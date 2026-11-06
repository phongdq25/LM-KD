"""LUFFY: Learning to Reason Under Off-Policy Guidance - Loss Criterion.

Paper: LUFFY: Learning to Reason Under Off-Policy Guidance (NeurIPS 2025)
Source: LUFFY/luffy/verl/verl/mix_src/mix_core_alg.py:69-251 & mix_actor.py:144-194

Key Method Characteristics:
1. Off-Policy Policy Shaping: Regularized Importance Sampling (e.g. p / (p + c)) or power/log-prob
   transformations on off-policy demonstration traces to emphasize low-probability yet critical reasoning steps.
2. Dual Token Loss Formulation:
   - On-policy tokens: Standard PPO surrogate objective with ratio clipping and optional reshaping.
   - Off-policy tokens: Reshaped importance ratio multiplied by group-relative advantages.
3. Flexible normalizations and clipping:
   - Optional removal of token mean (sequence length normalization as used in verl/LUFFY).
   - Optional removal of PPO clipping for unconstrained exploration.
   - Optional SFT multi-task loss branch for joint RL + SFT learning.
"""

from __future__ import annotations

import torch
from pydantic import Field

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.functional.rl import masked_mean
from lmkd.models.forward import FeatureRequest, SideRequest


def compute_sft_pure_loss(log_prob: torch.Tensor, eos_mask: torch.Tensor) -> torch.Tensor:
    """Compute pure SFT cross-entropy loss over response tokens (LUFFY mix_core_alg.py:7-10)."""
    sft_losses = -log_prob
    return masked_mean(sft_losses, eos_mask)


def compute_token_on_off_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float = 0.2,
    clip_upper_bound: float = 1.0,
    prefix_mask: torch.Tensor | None = None,
    off_cliprange: float = 0.2,
    off_normalize: bool = False,
    off_abs_cliprange: float | None = None,
    off_max_clip: float | None = None,
    off_min_clip: float | None = None,
    all_max_clip: float | None = None,
    off_policy_reshape: str = "p_div_p_0.1",
    off_policy_reshape_weight: float = 1.0,
    off_policy_reshape_pow_exp: float = 0.5,
    on_policy_reshape: str = "no_reshape",
    on_policy_reshape_weight: float = 1.0,
    on_policy_reshape_pow_exp: float = 0.5,
    target_probs: torch.Tensor | None = None,
    loss_remove_token_mean: bool = False,
    loss_remove_clip: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute LUFFY token-level on-policy and off-policy loss (LUFFY mix_core_alg.py:69-251).

    Args:
        old_log_prob: Student log-probabilities from rollout [B, T].
        log_prob: Student log-probabilities under current policy [B, T].
        advantages: Modulated advantage estimates [B, T] or [B].
        eos_mask: Mask of valid response tokens [B, T].
        cliprange: PPO clipping range for on-policy tokens.
        clip_upper_bound: Upper clipping bound.
        prefix_mask: Mask indicating off-policy prefix/demonstration tokens (1 for off-policy, 0 for on-policy).
        off_cliprange: Off-policy clip range.
        off_normalize: Normalize off-policy ratios.
        off_abs_cliprange: Optional absolute clip range.
        off_max_clip: Upper bound for off-policy ratio clipping.
        off_min_clip: Lower bound for off-policy ratio clipping.
        all_max_clip: Token probability threshold filter.
        off_policy_reshape: Policy shaping method for off-policy tokens
            ("no_reshape", "logp", "p_logp", "square_root", "pow", "p_div_p_0.1", "p_div_p_0.3", "p_div_p_0.5").
        off_policy_reshape_weight: Weight for linear logp terms.
        off_policy_reshape_pow_exp: Exponent for power transformation.
        on_policy_reshape: Reshaping method for on-policy tokens ("no_reshape", "logp", "p_logp", "square_root", "pow", "p_div_p_0.1", "p_div_p_0.5").
        on_policy_reshape_weight: Weight for on-policy linear logp terms.
        on_policy_reshape_pow_exp: Exponent for on-policy power transformation.
        target_probs: Optional reference teacher probabilities for importance sampling [B, T].
        loss_remove_token_mean: If True, normalize by sequence length T rather than valid token count.
        loss_remove_clip: If True, disable on-policy PPO clipping.

    Returns:
        dict containing 'pg_loss', 'off_pg_loss', 'on_pg_loss', 'ppo_kl', 'on_pg_clipfrac',
        'off_pg_clipfrac', 'off_policy_prob', 'on_policy_prob', 'off_ratio_mean', etc.
    """
    if prefix_mask is None:
        prefix_mask = torch.zeros_like(log_prob, dtype=torch.bool)
    else:
        prefix_mask = prefix_mask.bool()

    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1).expand_as(log_prob)

    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    # 1. On-policy ratio & reshaping
    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl)
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.sqrt(torch.exp(negative_approx_kl))
    elif on_policy_reshape == "pow":
        ratio = torch.pow(torch.exp(negative_approx_kl), on_policy_reshape_pow_exp)
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.5)
        f_old_prob = old_prob / (old_prob + 0.5)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    on_pg_losses = -advantages * ratio
    upper_bound = max(clip_upper_bound, 1.0 + cliprange)

    if not loss_remove_clip:
        on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
        on_pg_clipfrac = masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
    else:
        on_pg_loss = masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0, device=log_prob.device)

    # 2. Off-policy ratio & policy shaping
    if target_probs is None:
        off_ratio = torch.exp(log_prob)
        if off_policy_reshape == "no_reshape":
            pass
        elif off_policy_reshape == "logp":
            off_ratio = log_prob * off_policy_reshape_weight
        elif off_policy_reshape == "p_logp":
            off_ratio = log_prob * off_policy_reshape_weight + off_ratio
        elif off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif off_policy_reshape == "p_div_p_0.1":
            off_ratio = off_ratio / (off_ratio + 0.1)
        elif off_policy_reshape == "p_div_p_0.5":
            off_ratio = off_ratio / (off_ratio + 0.5)
        elif off_policy_reshape == "p_div_p_0.3":
            off_ratio = off_ratio / (off_ratio + 0.3)
        elif off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
        else:
            raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
    else:
        assert target_probs.shape == log_prob.shape
        off_ratio = torch.exp(log_prob) / (target_probs + 1e-6)
        off_ratio = off_ratio * prefix_mask

    # 3. Ratio clipping
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0, device=log_prob.device)

    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0, device=log_prob.device)

    off_ratio_mean = masked_mean(off_ratio, prefix_mask * eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0, device=log_prob.device)

    off_pg_losses = -advantages * off_ratio
    off_pg_loss = masked_mean(off_pg_losses, prefix_mask * eos_mask)
    if off_pg_loss.isnan().item():
        off_pg_loss = torch.tensor(0.0, device=log_prob.device)
    off_pg_clipfrac = torch.tensor(0.0, device=log_prob.device)

    # 4. Combine on-policy and off-policy losses
    prefix_mask_f = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask_f + on_pg_losses * (1.0 - prefix_mask_f)

    # 5. Probabilities tracking
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = masked_mean(off_policy_probs, prefix_mask * eos_mask)
    if off_policy_prob.isnan().item():
        off_policy_prob = torch.tensor(0.0, device=log_prob.device)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = masked_mean(on_policy_probs, (1.0 - prefix_mask_f) * eos_mask)
    if on_policy_prob.isnan().item():
        on_policy_prob = torch.tensor(0.0, device=log_prob.device)

    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask

    # 6. Global or token-mean reduction
    if loss_remove_token_mean:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
    else:
        pg_loss = masked_mean(pg_losses, eos_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }


def luffy_loss(
    logits: torch.Tensor,
    old_logp: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    prefix_mask: torch.Tensor | None = None,
    off_policy_reshape: str = "p_div_p_0.1",
    cliprange: float = 0.2,
    clip_upper_bound: float = 1.0,
    loss_remove_clip: bool = True,
    loss_remove_token_mean: bool = False,
) -> torch.Tensor:
    """Convenience functional interface for LUFFY loss."""
    logp = logits.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = labels.ne(-100)
    res = compute_token_on_off_policy_loss(
        old_log_prob=old_logp,
        log_prob=logp,
        advantages=advantages,
        eos_mask=mask,
        cliprange=cliprange,
        clip_upper_bound=clip_upper_bound,
        prefix_mask=prefix_mask,
        off_policy_reshape=off_policy_reshape,
        loss_remove_clip=loss_remove_clip,
        loss_remove_token_mean=loss_remove_token_mean,
    )
    return res["pg_loss"]


@CRITERIONS.register("luffy", source="LUFFY/luffy/verl/verl/mix_src/mix_core_alg.py:69-251", force=True)
@CRITERIONS.register("luffy_loss", source="LUFFY/luffy/verl/verl/mix_src/mix_core_alg.py:69-251", force=True)
class LUFFYLoss(Criterion):
    """LUFFY policy optimization criterion with regularized off-policy importance sampling.

    Paper: LUFFY: Learning to Reason Under Off-Policy Guidance (NeurIPS 2025)
    Source: LUFFY/luffy/verl/verl/mix_src/mix_core_alg.py:69-251
    """

    requires = ("student.labels", "rollout.advantage")
    supports_packing = False

    class Config(BaseConfig):
        cliprange: float = Field(default=0.2, ge=0, lt=1)
        clip_upper_bound: float = Field(default=1.0, ge=0)
        off_cliprange: float = Field(default=0.2, ge=0)
        off_policy_reshape: str = "p_div_p_0.1"
        off_policy_reshape_weight: float = 1.0
        off_policy_reshape_pow_exp: float = 0.5
        on_policy_reshape: str = "no_reshape"
        on_policy_reshape_weight: float = 1.0
        on_policy_reshape_pow_exp: float = 0.5
        off_max_clip: float | None = None
        off_min_clip: float | None = None
        all_max_clip: float | None = None
        loss_remove_token_mean: bool = False
        loss_remove_clip: bool = True
        temperature: float = 1.0
        use_sft_multitask_loss: bool = False
        sft_loss_coef: float = 1.0

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx: LossContext) -> LossOutput:
        s_view = ctx.student_logits()
        if self.cfg.temperature != 1.0:
            s_view = s_view.with_temperature(self.cfg.temperature)
        logp = s_view.target_logp(ctx.batch["student.labels"])
        old_logp = ctx.batch.get("rollout.old_logp", logp.detach())
        adv = ctx.batch["rollout.advantage"]
        prefix_mask = ctx.batch.get("rollout.prefix_mask", ctx.batch.get("prefix_mask", None))
        if prefix_mask is None:
            prefix_mask = torch.zeros_like(logp, dtype=torch.bool)
        else:
            prefix_mask = prefix_mask.bool()
        mask = ctx.supervised_mask()
        target_probs = ctx.batch.get("target_probs", None)

        if self.cfg.use_sft_multitask_loss:
            off_policy_mask = prefix_mask.any(-1) if prefix_mask.ndim > 1 else prefix_mask
            off_logp = logp[off_policy_mask]
            off_mask = mask[off_policy_mask]
            if off_logp.numel() > 0:
                sft_loss = compute_sft_pure_loss(off_logp, off_mask)
            else:
                sft_loss = torch.tensor(0.0, device=logp.device)

            on_policy_mask = ~off_policy_mask
            on_logp = logp[on_policy_mask]
            on_old_logp = old_logp[on_policy_mask]
            on_adv = adv[on_policy_mask]
            on_mask = mask[on_policy_mask]

            dummy_prefix = torch.zeros_like(on_logp, dtype=torch.bool)
            on_res = compute_token_on_off_policy_loss(
                old_log_prob=on_old_logp,
                log_prob=on_logp,
                advantages=on_adv,
                eos_mask=on_mask,
                cliprange=self.cfg.cliprange,
                clip_upper_bound=self.cfg.clip_upper_bound,
                prefix_mask=dummy_prefix,
                off_cliprange=self.cfg.off_cliprange,
                off_policy_reshape=self.cfg.off_policy_reshape,
                on_policy_reshape=self.cfg.on_policy_reshape,
                loss_remove_token_mean=self.cfg.loss_remove_token_mean,
                loss_remove_clip=self.cfg.loss_remove_clip,
            )
            on_pg_loss = on_res["pg_loss"]
            total_loss = sft_loss * self.cfg.sft_loss_coef + on_pg_loss
            return LossOutput(
                {"pg": LossTerm(total_loss, "token_sum" if self.cfg.loss_remove_token_mean else "mean")},
                metrics={
                    "sft_loss": sft_loss.detach(),
                    "on_pg_loss": on_pg_loss.detach(),
                    "approx_kl": on_res["ppo_kl"].detach(),
                },
            )

        res = compute_token_on_off_policy_loss(
            old_log_prob=old_logp,
            log_prob=logp,
            advantages=adv,
            eos_mask=mask,
            cliprange=self.cfg.cliprange,
            clip_upper_bound=self.cfg.clip_upper_bound,
            prefix_mask=prefix_mask,
            off_cliprange=self.cfg.off_cliprange,
            off_max_clip=self.cfg.off_max_clip,
            off_min_clip=self.cfg.off_min_clip,
            all_max_clip=self.cfg.all_max_clip,
            off_policy_reshape=self.cfg.off_policy_reshape,
            off_policy_reshape_weight=self.cfg.off_policy_reshape_weight,
            off_policy_reshape_pow_exp=self.cfg.off_policy_reshape_pow_exp,
            on_policy_reshape=self.cfg.on_policy_reshape,
            on_policy_reshape_weight=self.cfg.on_policy_reshape_weight,
            on_policy_reshape_pow_exp=self.cfg.on_policy_reshape_pow_exp,
            target_probs=target_probs,
            loss_remove_token_mean=self.cfg.loss_remove_token_mean,
            loss_remove_clip=self.cfg.loss_remove_clip,
        )
        return LossOutput(
            {"pg": LossTerm(res["pg_loss"], "token_sum" if self.cfg.loss_remove_token_mean else "mean")},
            metrics={
                "off_pg_loss": res["off_pg_loss"].detach(),
                "on_pg_loss": res["on_pg_loss"].detach(),
                "off_pg_clipfrac": res["off_pg_clipfrac"].detach(),
                "on_pg_clipfrac": res["on_pg_clipfrac"].detach(),
                "approx_kl": res["ppo_kl"].detach(),
                "off_policy_prob": res["off_policy_prob"].detach(),
                "on_policy_prob": res["on_policy_prob"].detach(),
                "off_ratio_mean": res["off_ratio_mean"].detach(),
            },
        )


__all__ = [
    "LUFFYLoss",
    "compute_token_on_off_policy_loss",
    "compute_sft_pure_loss",
    "luffy_loss",
]
