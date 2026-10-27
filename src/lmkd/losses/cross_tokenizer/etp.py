"""Entropic transport and semi-dual MCW transport from IMPACT ETP.py / ETP_Dual.py."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ETP(nn.Module):
    """Sinkhorn transport with the original IMPACT update order, evaluated in fp32."""

    def __init__(self, sinkhorn_alpha: float = 0.1, stop_thr: float = 1e-9,
                 ot_max_iter: int = 100, epsilon: float = 1e-9) -> None:
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.stop_thr = stop_thr
        self.ot_max_iter = ot_max_iter
        self.epsilon = epsilon

    def forward(self, cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = cost.float()
        a = matrix.new_ones(matrix.shape[0], 1) / matrix.shape[0]
        b = matrix.new_ones(matrix.shape[1], 1) / matrix.shape[1]
        u = torch.ones_like(a) / a.shape[0]
        kernel = torch.exp(-matrix * self.sinkhorn_alpha)
        err: float | torch.Tensor = 1.0
        count = 0
        while err > self.stop_thr and count < self.ot_max_iter:
            v = b / (kernel.T @ u + self.epsilon)
            u = a / (kernel @ v + self.epsilon)
            count += 1
            if count % 50 == 1:
                bb = v * (kernel.T @ u)
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float("inf"))
        transport = u * (kernel * v.T)
        return torch.sum(transport * matrix), transport


class DualPotentialNetwork(nn.Module):
    """Context-aware potential φ from IMPACT ETP_Dual.py:8-53."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 1,
                 use_context: bool = True) -> None:
        super().__init__()
        self.use_context = use_context
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, output_dim))
        if use_context:
            self.context_proj = nn.Linear(input_dim, hidden_dim)
            self.token_proj = nn.Linear(input_dim, hidden_dim)
            self.combine = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if not self.use_context or context is None:
            return torch.tanh(self.network(x).squeeze(-1))
        if x.dim() == 2:
            context_features = self.context_proj(context).unsqueeze(0).expand(x.size(0), -1)
        else:
            context_features = self.context_proj(context).unsqueeze(1).expand(-1, x.size(1), -1)
        features = torch.cat([context_features, self.token_proj(x)], dim=-1)
        return torch.tanh(self.combine(features).squeeze(-1))


class MultiCostSemiDualOT(nn.Module):
    """IMPACT multi-cost semi-dual OT with trainable β and in-forward φ ascent."""

    def __init__(self, num_costs: int, input_dim_x: int, input_dim_y: int,
                 hidden_dim: int = 256, epsilon: float = 0.1,
                 num_inner_iters: int = 4, inner_lr: float = 0.01) -> None:
        super().__init__()
        self.num_costs = num_costs
        self.epsilon = epsilon
        self.phi = DualPotentialNetwork(input_dim_x, hidden_dim)
        self.beta = nn.Parameter(torch.zeros(num_costs))
        self.num_inner_iters = num_inner_iters
        self.inner_lr = inner_lr
        self.proj_y = nn.Linear(input_dim_y, input_dim_x) if input_dim_x != input_dim_y else nn.Identity()

    @property
    def alpha(self) -> torch.Tensor:
        """The learned simplex weights for the cost matrices."""
        return F.softmax(self.beta, dim=0)

    def compute_c_transform_multi(self, phi_x: torch.Tensor, costs: list[torch.Tensor],
                                  mu: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        transformed = []
        for cost in costs:
            exponent = ((phi_x.clamp(-50, 50).unsqueeze(1) - cost) / self.epsilon).clamp(-50, 50)
            exponent = exponent + torch.log(mu + 1e-10).unsqueeze(1)
            phi_c = -self.epsilon * torch.logsumexp(exponent, dim=0)
            transformed.append(torch.where(torch.isfinite(phi_c), phi_c, torch.zeros_like(phi_c)))
        weighted = (self.alpha[:, None] * torch.stack(transformed)).sum(0)
        return weighted, transformed

    def _ascent(self, x: torch.Tensor, costs: list[torch.Tensor], mu: torch.Tensor,
                nu: torch.Tensor, context: torch.Tensor) -> None:
        for _ in range(self.num_inner_iters):
            phi_x = self.phi(x, context)
            if torch.isnan(phi_x).any():
                break
            phi_c, _ = self.compute_c_transform_multi(phi_x, costs, mu)
            if torch.isnan(phi_c).any():
                break
            wasserstein = torch.dot(phi_x, mu) + torch.dot(phi_c, nu)
            if torch.isnan(wasserstein):
                break
            if self.training:
                try:
                    grads = torch.autograd.grad(-wasserstein, tuple(self.phi.parameters()),
                                                retain_graph=True, allow_unused=True)
                    with torch.no_grad():
                        for param, grad in zip(self.phi.parameters(), grads):
                            if grad is not None:
                                param.data = param.data - self.inner_lr * grad.clamp(-1, 1)
                except RuntimeError:
                    break

    def forward(self, x: torch.Tensor, y: torch.Tensor, costs: list[torch.Tensor],
                mu: torch.Tensor | None = None,
                nu: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(costs) != self.num_costs:
            raise ValueError("cost count does not match num_costs")
        mu = mu if mu is not None else x.new_ones(x.shape[0]) / x.shape[0]
        nu = nu if nu is not None else y.new_ones(y.shape[0]) / y.shape[0]
        context = x.mean(0)
        self._ascent(x, costs, mu, nu, context)
        phi_x = self.phi(x, context)
        phi_x = torch.where(torch.isfinite(phi_x), phi_x, torch.zeros_like(phi_x))
        phi_c, each = self.compute_c_transform_multi(phi_x, costs, mu)
        common = torch.dot(phi_x, mu)
        loss = common + torch.dot(phi_c, nu)
        loss = torch.where(torch.isfinite(loss), loss, loss.new_zeros(()))
        terms = [torch.where(torch.isfinite(w), w, w.new_zeros(()))
                 for w in (common + torch.dot(c, nu) for c in each)]
        return loss, terms
