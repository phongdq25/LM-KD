"""MCW-KD multi-cost OT over positional logits and DTW-aligned hidden states."""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo, PrepareContext
from lmkd.losses.cross_tokenizer.etp import ETP, MultiCostSemiDualOT
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import DSLProjector, ModuleRegistry
from lmkd.teachers.signals import TeacherSignalSpec

LOG = logging.getLogger(__name__)


def context_windows(seq: torch.Tensor, window: int) -> torch.Tensor:
    """Mean each inclusive context window using a vectorized prefix sum."""
    size = seq.shape[0]
    prefix = torch.cat([seq.new_zeros((1, seq.shape[-1])), seq.cumsum(0)], dim=0)
    pos = torch.arange(size, device=seq.device)
    starts = (pos - window).clamp(min=0)
    ends = (pos + window + 1).clamp(max=size)
    return (prefix[ends] - prefix[starts]) / (ends - starts).unsqueeze(-1)


def _topk_positional(view: LogitsView, k: int, chunk_size: int = 128) -> torch.Tensor:
    batch, length = view.shape_bt
    chunks = []
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        rows = torch.arange(batch, device=view.device).repeat_interleave(end - start)
        pos = torch.arange(start + 1, end + 1, device=view.device).repeat(batch)
        logits = view.at(torch.stack((rows, pos), dim=-1))
        chunks.append(logits.topk(min(k, logits.shape[-1]), dim=-1).values.reshape(batch, end - start, -1))
    return torch.cat(chunks, dim=1)


def _pair_matrix(pairs: Any, values: Any, rows: int, cols: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros((rows, cols), device=device)
    p = torch.as_tensor(pairs, device=device, dtype=torch.long).reshape(-1, 2)
    v = torch.as_tensor(values, device=device, dtype=torch.float32).reshape(-1)
    valid = (p[:, 0] >= 0) & (p[:, 0] < rows) & (p[:, 1] >= 0) & (p[:, 1] < cols)
    out[p[valid, 0], p[valid, 1]] = v[valid]
    return out


def _normalize_cost(cost: torch.Tensor) -> torch.Tensor:
    maximum = cost.max()
    scaled = cost / (maximum + 1e-7) if maximum > 1e-7 else cost
    return torch.where(torch.isfinite(scaled), scaled, torch.zeros_like(scaled))


def _get_or_add(modules: ModuleRegistry, name: str, factory: Any, group: str, owner: str) -> nn.Module:
    return modules.get(name) if name in modules else modules.add(name, factory(), group=group, owner=owner)


class _OTConfig(BaseConfig):
    top_k_vocab: int = 300
    window: int = 4
    dual: bool = False
    tau_seq: float = 2.0
    t_start: float = 0.1
    t_end: float = 1.0
    cost_weights_init: list[float] = [0.2, 0.5, 0.3]
    sinkhorn_alpha: float = 0.1
    ot_max_iter: int = 100
    stop_thr: float = 1e-9
    ot_weight: float = 1.0
    update_cost_weights: bool = True
    update_interval: int = 1000
    sigma: float = 2.0
    epsilon: float = 0.1
    inner_iters: int = 4
    inner_lr: float = 0.01
    dual_hidden: int = 256
    shared_prefix: str = "mcw"
    compat_untrained_aux: bool = False
    legacy_scale: bool = False


class _OTBase(Criterion):
    """Shared MCW module ownership, Sinkhorn or semi-dual solving, and delayed QP updates."""

    needs_teacher: ClassVar[bool] = True
    cost_w: torch.Tensor

    def __init__(self, cfg: BaseConfig | None = None) -> None:
        super().__init__(cfg)
        self.register_buffer("cost_w", torch.tensor(self.cfg.cost_weights_init, dtype=torch.float32))
        self._cost_buf: list[torch.Tensor] = []
        self._etp = ETP(self.cfg.sinkhorn_alpha, self.cfg.stop_thr, self.cfg.ot_max_iter)

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view", hidden=(-1,)),
                              teacher=SideRequest(logits="view", hidden=(-1,)),
                              teacher_signal=TeacherSignalSpec(kind="full"))

    def _group(self) -> str:
        return "mcw_untrained" if self.cfg.compat_untrained_aux else "projector"

    def _build_ot(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        if info.teacher_hidden is None:
            raise ValueError("MCW-KD needs a teacher hidden size")
        cfg = DSLProjector.Config(structure="t-s", s=info.student_hidden,
                                  t=info.teacher_hidden, bias=True)
        _get_or_add(modules, f"{self.cfg.shared_prefix}__ot", lambda: DSLProjector(cfg),
                    self._group(), self.alias)

    def _build_dual(self, modules: ModuleRegistry, suffix: str, dim: int) -> None:
        _get_or_add(modules, f"{self.cfg.shared_prefix}__dual_{suffix}",
                    lambda: MultiCostSemiDualOT(3, dim, dim, self.cfg.dual_hidden,
                                                self.cfg.epsilon, self.cfg.inner_iters,
                                                self.cfg.inner_lr), self._group(), self.alias)

    def prepare(self, ctx: PrepareContext) -> None:
        if self.cfg.update_cost_weights and not self.cfg.dual:
            try:
                import cvxpy  # noqa: F401
            except ImportError as exc:
                raise ImportError('MCW QP needs cvxpy: pip install -e ".[ot]"') from exc

    def _solve(self, costs: list[torch.Tensor], x: torch.Tensor, y: torch.Tensor,
               ctx: LossContext, suffix: str) -> torch.Tensor:
        if self.cfg.dual:
            dual: MultiCostSemiDualOT = ctx.modules.get(f"{self.cfg.shared_prefix}__dual_{suffix}")
            loss, values = dual(x.float(), y.float(), costs)
            self._cost_buf.append(torch.stack([v.detach() for v in values]))
            return loss
        total = sum(w * cost for w, cost in zip(self.cost_w, costs))
        loss, _ = self._etp(total)
        self._cost_buf.append(torch.stack([c.detach().mean() for c in costs]))
        return loss

    def _output(self, losses: list[torch.Tensor], ctx: LossContext, name: str) -> LossOutput:
        value = torch.stack(losses).sum() if losses else ctx.student.hidden(-1).sum() * 0
        value = self.cfg.ot_weight * value
        if self.cfg.legacy_scale:
            value = value / max(1, ctx.batch["student.attention_mask"].shape[0])
        reduction: Literal["token_sum", "seq_sum"] = "token_sum" if self.cfg.legacy_scale else "seq_sum"
        return LossOutput({name: LossTerm(value, reduction)})

    def on_step_end(self, state: Any) -> None:
        if not self.cfg.update_cost_weights or self.cfg.dual:
            self._cost_buf.clear()
            return
        if state.step % self.cfg.update_interval or not self._cost_buf:
            return
        import cvxpy as cp

        mean_cost = torch.stack(self._cost_buf).mean(0).cpu().numpy()
        self._cost_buf.clear()
        count = len(mean_cost)
        weights = cp.Variable(count)
        objective = cp.Minimize(mean_cost @ weights + self.cfg.sigma * cp.sum_squares(weights - 1 / count))
        problem = cp.Problem(objective, [cp.sum(weights) == 1, weights >= 0.01])
        try:
            problem.solve()
            if weights.value is not None:
                self.cost_w.copy_(torch.as_tensor(weights.value, device=self.cost_w.device, dtype=self.cost_w.dtype))
            else:
                LOG.warning("MCW QP returned no weights; retaining previous cost weights")
        except cp.error.SolverError as exc:
            LOG.warning("MCW QP failed; retaining previous cost weights: %s", exc)

    def extra_state(self) -> dict[str, Any]:
        return {"cost_w": self.cost_w.detach().cpu()}

    def load_extra_state(self, state: dict[str, Any]) -> None:
        if "cost_w" in state:
            self.cost_w.copy_(state["cost_w"].to(self.cost_w.device))


@CRITERIONS.register("ot_logits", source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/MCW_KD.py:118-176")
class OTLogits(_OTBase):
    """MCW OT on sorted positional logits; cross-tokenizer interpolation is positional only and has no linguistic meaning.

    Source bugs repaired: ``ot_weight`` supplies a missing attribute, cost weights are buffers, and
    auxiliary modules train by default. ``compat_untrained_aux`` assigns a zero-lr recipe group;
    ``update_cost_weights`` repairs the absent QP call; ``legacy_scale`` restores the extra batch divisor.
    """

    class Config(_OTConfig):
        """OT logits settings, including source-compatible cost and transport controls."""

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        self._build_ot(info, modules)
        _get_or_add(modules, f"{self.cfg.shared_prefix}__salience",
                    lambda: nn.Linear(info.student_hidden, 1, bias=True), self._group(), self.alias)
        if self.cfg.dual:
            width = min(self.cfg.top_k_vocab, info.student_vocab, info.teacher_vocab or info.student_vocab)
            self._build_dual(modules, "logits", width)

    def _costs(self, sp: torch.Tensor, tp: torch.Tensor, hs: torch.Tensor, ht: torch.Tensor,
               salience: nn.Module) -> list[torch.Tensor]:
        if self.cfg.dual:
            sp, tp = sp.clamp(1e-7, 1), tp.clamp(1e-7, 1)
        c2 = torch.cdist(tp, sp)
        c4 = (tp[:, None] * torch.log(tp[:, None] / (sp[None] + 1e-7))).sum(-1)
        salience_dtype = next(salience.parameters()).dtype
        ss = torch.sigmoid(salience(hs.to(salience_dtype))).squeeze(-1)
        st = torch.sigmoid(salience(ht.to(salience_dtype))).squeeze(-1)
        cs = (st[:, None] - ss[None]).abs()
        if self.cfg.dual:
            return [_normalize_cost(c2), _normalize_cost(c4.clamp(0, 100)), _normalize_cost(cs)]
        return [c2, c4, cs]

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.side is None or ctx.teacher.logits is None:
            raise ValueError("ot_logits needs teacher logits and hidden states")
        sv = ctx.student_logits()
        tv = ctx.teacher.logits
        s = _topk_positional(sv, self.cfg.top_k_vocab)
        with torch.no_grad():
            t = _topk_positional(tv, self.cfg.top_k_vocab)
        width = min(s.shape[-1], t.shape[-1])
        s, t = s[..., :width], t[..., :width]
        length = max(s.shape[1], t.shape[1])
        s = F.pad(s, (0, 0, 0, length - s.shape[1]))
        t = F.pad(t, (0, 0, 0, length - t.shape[1]))
        progress = min(ctx.state.step / max(ctx.state.total_steps, 1), 1)
        factor = self.cfg.t_start + (self.cfg.t_end - self.cfg.t_start) * progress
        sp = F.softmax(s / self.cfg.tau_seq, dim=-1)
        tp = F.softmax(((1 - factor) * s + factor * t) / self.cfg.tau_seq, dim=-1)
        h_s = ctx.student.hidden(-1)
        projector = ctx.modules.get(f"{self.cfg.shared_prefix}__ot")
        h_t = projector(ctx.teacher.side.hidden(-1).to(next(projector.parameters()).dtype))
        salience = ctx.modules.get(f"{self.cfg.shared_prefix}__salience")
        mask_s = ctx.batch["student.attention_mask"].bool()
        mask_t = ctx.batch["teacher.attention_mask"].bool()
        losses = []
        for b in range(s.shape[0]):
            if not mask_s[b].any() or not mask_t[b].any():
                continue
            x, y = tp[b, :mask_t.shape[1]][mask_t[b]], sp[b, :mask_s.shape[1]][mask_s[b]]
            costs = self._costs(y, x, h_s[b][mask_s[b]], h_t[b][mask_t[b]], salience)
            losses.append(self._solve(costs, x, y, ctx, "logits"))
        return self._output(losses, ctx, "ot_logits")


@CRITERIONS.register("ot_hidden", source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/MCW_KD.py:178-257")
class OTHidden(_OTBase):
    """MCW OT on hidden context vectors and full-sequence DTW edit paths.

    Source bugs repaired: ``ot_weight`` supplies a missing attribute and the overwritten scaling
    line is removed. ``update_cost_weights`` repairs the missing QP call; ``legacy_scale`` restores
    the source divisor; ``compat_untrained_aux`` moves shared projectors to a zero-lr recipe group.
    """

    requires: ClassVar[tuple[str, ...]] = ("align.dtw_s2t", "align.dtw_s2t_cost",
                                           "align.dtw_t2s", "align.dtw_t2s_cost")

    class Config(_OTConfig):
        """OT hidden settings, with source cost initialization."""
        cost_weights_init: list[float] = [0.3, 0.5, 0.2]

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        self._build_ot(info, modules)
        if self.cfg.dual:
            self._build_dual(modules, "hidden", info.student_hidden)

    def _costs(self, s: torch.Tensor, t: torch.Tensor, ctx: LossContext, b: int) -> list[torch.Tensor]:
        n, m = s.shape[0], t.shape[0]
        st = _pair_matrix(ctx.batch["align.dtw_s2t"][b], ctx.batch["align.dtw_s2t_cost"][b], n, m, s.device)
        ts = _pair_matrix(ctx.batch["align.dtw_t2s"][b], ctx.batch["align.dtw_t2s_cost"][b], m, n, s.device)
        c2 = (st.T + ts) / 2
        if self.cfg.dual:
            s, t = F.normalize(s.float(), dim=-1), F.normalize(t.float(), dim=-1)
            c2 = _normalize_cost(c2)
        cs, ct = context_windows(s, self.cfg.window), context_windows(t, self.cfg.window)
        c5 = _normalize_cost(torch.cdist(ct, cs))
        if self.cfg.dual:
            c6 = (1 - F.normalize(ct, dim=-1) @ F.normalize(cs, dim=-1).T).clamp(0, 2)
            return [c2, c5, c6]
        nt = ct / (ct.norm(dim=-1, keepdim=True) + 1e-7)
        ns = cs / (cs.norm(dim=-1, keepdim=True) + 1e-7)
        return [c2, c5, 1 - nt @ ns.T]

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.side is None:
            raise ValueError("ot_hidden needs teacher hidden states")
        hs = ctx.student.hidden(-1).float()
        projector = ctx.modules.get(f"{self.cfg.shared_prefix}__ot")
        ht = projector(ctx.teacher.side.hidden(-1).to(next(projector.parameters()).dtype)).float()
        sm = ctx.batch["student.attention_mask"].bool()
        tm = ctx.batch["teacher.attention_mask"].bool()
        losses = []
        for b in range(hs.shape[0]):
            s, t = hs[b][sm[b]], ht[b][tm[b]]
            if not len(s) or not len(t):
                continue
            costs = self._costs(s, t, ctx, b)
            x, y = (F.normalize(t, dim=-1), F.normalize(s, dim=-1)) if self.cfg.dual else (t, s)
            losses.append(self._solve(costs, x, y, ctx, "hidden"))
        return self._output(losses, ctx, "ot_hidden")
