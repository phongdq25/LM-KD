"""Independent fp32 reference from IMPACT criterions/ETP.py:36-56."""
import torch
import torch.nn.functional as F


def etp_ref(cost, alpha=0.1, stop_thr=1e-9, max_iter=100, eps=1e-9):
    a = cost.new_ones(cost.shape[0], 1) / cost.shape[0]
    b = cost.new_ones(cost.shape[1], 1) / cost.shape[1]
    u = torch.ones_like(a) / a.shape[0]
    kernel = torch.exp(-cost * alpha)
    err = 1.0
    count = 0
    while err > stop_thr and count < max_iter:
        v = b / (kernel.T @ u + eps)
        u = a / (kernel @ v + eps)
        count += 1
        if count % 50 == 1:
            bb = v * (kernel.T @ u)
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float("inf"))
    transport = u * (kernel * v.T)
    return torch.sum(transport * cost), transport


def dual_ref(module, x, y, costs):
    """Explicit IMPACT ETP_Dual.py:204-314 equations with source φ ascent."""
    mu = x.new_ones(len(x)) / len(x)
    nu = y.new_ones(len(y)) / len(y)
    context = x.mean(0)

    def transform(phi):
        values = []
        for cost in costs:
            exponent = ((phi.clamp(-50, 50)[:, None] - cost) / module.epsilon).clamp(-50, 50)
            exponent = exponent + (mu + 1e-10).log()[:, None]
            item = -module.epsilon * torch.logsumexp(exponent, dim=0)
            values.append(torch.where(torch.isfinite(item), item, torch.zeros_like(item)))
        return (F.softmax(module.beta, 0)[:, None] * torch.stack(values)).sum(0), values

    for _ in range(module.num_inner_iters):
        potential = module.phi(x, context)
        mixed, _ = transform(potential)
        objective = potential @ mu + mixed @ nu
        grads = torch.autograd.grad(-objective, tuple(module.phi.parameters()), retain_graph=True,
                                    allow_unused=True)
        with torch.no_grad():
            for parameter, gradient in zip(module.phi.parameters(), grads):
                if gradient is not None:
                    parameter.data = parameter.data - module.inner_lr * gradient.clamp(-1, 1)
    potential = module.phi(x, context)
    mixed, each = transform(potential)
    return potential @ mu + mixed @ nu, [potential @ mu + item @ nu for item in each]
