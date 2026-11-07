"""Reference implementation of various divergence methods."""
from types import SimpleNamespace

import torch


class VariousDivergenceRef:
    def __init__(self, kd_temp=1.0, tea_temp=1.0, skew_lambda=0.1, adaptive_kl_alpha=0.5, padding_id=-100):
        self.kd_temp = kd_temp
        self.tea_temp = tea_temp
        self.padding_id = padding_id
        self.args = SimpleNamespace(skew_lambda=skew_lambda, adaptive_kl_alpha=adaptive_kl_alpha)

    def compute_forward_kl_divergence(
        self,
        logits,
        teacher_logits,
        target,
        reduction="sum",
        log=None,
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - lprobs))
        inf_mask = logits.isinf() | logits.isnan() | teacher_logits.isinf() | teacher_logits.isnan()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["forward_kl"] = kld

        return kld

    def compute_reverse_kl_divergence(
        self,
        logits,
        teacher_logits,
        target,
        reduction="sum",
        log=None,
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (probs * (lprobs - teacher_lprobs))
        inf_mask = logits.isinf() | logits.isnan() | teacher_logits.isinf() | teacher_logits.isnan()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["reverse_kl"] = kld

        return kld

    def compute_adaptive_kl_divergence(
        self,
        logits,
        teacher_logits,
        target,
        reduction="sum",
        log=None,
        use_tea_temp=False
    ):
        alpha = self.args.adaptive_kl_alpha
        probs = torch.softmax(
            logits / self.kd_temp, dim=-1, dtype=torch.float32
        )
        if use_tea_temp:
            teacher_probs = torch.softmax(
                teacher_logits / self.tea_temp / self.kd_temp, dim=-1, dtype=torch.float32
            )
        else:
            teacher_probs = torch.softmax(
                teacher_logits / self.kd_temp, dim=-1, dtype=torch.float32
            )
        sorted_teacher_probs, sorted_idx = teacher_probs.sort(-1)
        sorted_probs = probs.gather(-1, sorted_idx)
        gap = (sorted_teacher_probs - sorted_probs).abs()
        cum_teacher_probs = torch.cumsum(sorted_teacher_probs, -1)
        tail_mask = cum_teacher_probs.le(alpha).float()
        g_head = (gap * (1 - tail_mask)).sum(-1).detach()
        g_tail = (gap * tail_mask).sum(-1).detach()

        fkl = self.compute_forward_kl_divergence(logits, teacher_logits, target, reduction="none", use_tea_temp=use_tea_temp)
        rkl = self.compute_reverse_kl_divergence(logits, teacher_logits, target, reduction="none", use_tea_temp=use_tea_temp)

        akl = (g_head / (g_head + g_tail)) * fkl + (g_tail / (g_head + g_tail)) * rkl

        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            akl = akl.masked_fill_(pad_mask, 0.0)
            akl = akl.sum()

            if log is not None:
                log["adaptive_kl"] = akl

        return akl

    def compute_skewed_forward_kl_divergence(
        self,
        logits,
        teacher_logits,
        target,
        reduction="sum",
        log=None,
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = self.args.skew_lambda * teacher_probs + (1 - self.args.skew_lambda) * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | logits.isnan() | teacher_logits.isinf() | teacher_logits.isnan()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["skewed_forward_kl"] = kld

        return kld

    def compute_skewed_reverse_kl_divergence(
        self,
        logits,
        teacher_logits,
        target,
        reduction="sum",
        log=None,
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = (1 - self.args.skew_lambda) * teacher_probs + self.args.skew_lambda * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        student_lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        # teacher_lprobs = torch.log_softmax(teacher_logits / self.tea_temp / self.kd_temp, -1, dtype=torch.float32)
        kld = (student_probs * (student_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | logits.isnan() | teacher_logits.isinf() | teacher_logits.isnan()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["skewed_reverse_kl"] = kld

        return kld

    def compute_js_divergence(
        self,
        logits,
        teacher_logits,
        target,
        reduction="sum",
        log=None,
        use_tea_temp=False
    ):
        # temperature scaling
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        m_probs = (probs + teacher_probs) / 2

        lprobs = torch.log(probs + 1e-9)
        teacher_lprobs = torch.log(teacher_probs + 1e-9)
        m_lprobs = torch.log(m_probs + 1e-9)

        kld1 = teacher_probs * (teacher_lprobs - m_lprobs)
        kld2 = probs * (lprobs - m_lprobs)
        kld = (kld1 + kld2) / 2
        inf_mask = logits.isinf() | logits.isnan() | teacher_logits.isinf() | teacher_logits.isnan()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["js_div"] = kld

        return kld
