"""TeacherInputBuilder: what the teacher sees (DESIGN §5.6)."""
from __future__ import annotations

import re
from typing import Any

import torch
from pydantic import Field

from lmkd.core.batch import FIELDS, Batch, FieldSpec
from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import ALIGNERS, TEACHER_INPUTS
from lmkd.data.labels import make_labels
from lmkd.data.transforms.tokenize import render_prompt
from lmkd.rollout.base import SamplingSpec

FIELDS.register(FieldSpec("rollout.has_signal", "scalar", torch.float32))


def _truncate_prompt(p_ids: list[int], max_len: int, side: str) -> list[int]:
    """Truncate prompt token ids if longer than max_len."""
    if len(p_ids) <= max_len:
        return p_ids
    if side == "left":
        return p_ids[-max_len:]
    elif side == "right":
        return p_ids[:max_len]
    raise ValueError(f"unknown truncation_side {side!r}")


def _collate_teacher_prompts_and_responses(
    batch: Batch,
    rt: Any,
    tok: Any,
    teacher_prompts: list[list[int]],
) -> Batch:
    """Combine teacher prompts with student response tokens into padded teacher.* batch keys."""
    batch_size = batch.batch_size
    device = batch["student.input_ids"].device

    all_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    t_prompt_lens: list[int] = []

    s_plen = batch["student.prompt_len"].cpu().tolist() if "student.prompt_len" in batch else None
    s_mask = batch["student.attention_mask"].cpu()
    s_ids = batch["student.input_ids"].cpu()
    lengths = s_mask.sum(dim=-1).tolist()

    for b in range(batch_size):
        sp_b = int(s_plen[b]) if s_plen is not None else 0
        l_b = int(lengths[b])
        resp_b = s_ids[b, sp_b:l_b].tolist()

        p_ids = teacher_prompts[b]
        tp_b = len(p_ids)
        t_prompt_lens.append(tp_b)

        ids = p_ids + resp_b
        labels = make_labels(ids, tp_b).tolist()

        all_ids.append(ids)
        all_labels.append(labels)

    max_len = max(len(row) for row in all_ids)
    pad_mult = getattr(getattr(rt.config, "data", None), "pad_to_multiple_of", None)
    if pad_mult is not None and pad_mult > 1:
        rem = max_len % pad_mult
        if rem != 0:
            max_len += (pad_mult - rem)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    t_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    t_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    t_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        row_len = len(all_ids[b])
        t_input_ids[b, :row_len] = torch.tensor(all_ids[b], dtype=torch.long, device=device)
        t_attention_mask[b, :row_len] = 1
        t_labels[b, :row_len] = torch.tensor(all_labels[b], dtype=torch.long, device=device)

    batch["teacher.input_ids"] = t_input_ids
    batch["teacher.attention_mask"] = t_attention_mask
    batch["teacher.labels"] = t_labels
    batch["teacher.prompt_len"] = torch.tensor(t_prompt_lens, dtype=torch.long, device=device)

    return batch


class TeacherInputBuilder(Plugin):
    """Builds the ``teacher.*`` batch keys from the student batch (same input, retokenized, privileged prompt,
    ...).
    """
    def setup(self, rt: Any) -> None:
        self.rt = rt

    def build(self, batch: Batch, rt: Any) -> Batch:
        raise NotImplementedError


@TEACHER_INPUTS.register("same")
class SameInput(TeacherInputBuilder):
    """Teacher reads the student inputs (same tokenizer, same prompt). Nothing to build."""

    class Config(BaseConfig):
        pass

    def build(self, batch, rt):
        return batch


@TEACHER_INPUTS.register("pretokenized")
class PretokenizedInput(TeacherInputBuilder):
    """``teacher.*`` already produced by a transform (e.g. tokenize_sft(teacher=true))."""

    requires = ("teacher.input_ids", "teacher.attention_mask")

    class Config(BaseConfig):
        pass

    def build(self, batch, rt):
        batch.require(*self.requires)
        return batch


@TEACHER_INPUTS.register("privileged_prompt", source="Self-Distill/OPSD-main/data_collator.py:60-110")
class PrivilegedPrompt(TeacherInputBuilder):
    """Teacher sees privileged context (e.g. problem + reference solution) followed by student response tokens."""

    provides = ("teacher.input_ids", "teacher.attention_mask", "teacher.labels", "teacher.prompt_len")

    class Config(BaseConfig):
        template: str
        fields: list[str] = ["solution"]
        use_chat_template: bool = True
        chat_template_kwargs: dict = {}
        max_prompt_len: int = 4096
        truncation_side: str = "left"
        tokenizer: str = "student"

    def build(self, batch: Batch, rt: Any) -> Batch:
        tok = (
            rt.student_tokenizer
            if self.cfg.tokenizer == "student"
            else (rt.teacher_tokenizer or rt.student_tokenizer)
        )
        batch_size = batch.batch_size
        teacher_prompts: list[list[int]] = []

        for b in range(batch_size):
            meta_b = {k.removeprefix("meta."): v[b] for k, v in batch.meta.items() if k.startswith("meta.")}
            for f in self.cfg.fields:
                if f not in meta_b:
                    raise KeyError(
                        f"PrivilegedPrompt: required meta field {f!r} missing. Available meta: {sorted(meta_b)}"
                    )
            text_b = self.cfg.template.format(**meta_b)
            p_text = render_prompt(tok, text_b, self.cfg.use_chat_template, **self.cfg.chat_template_kwargs)
            p_ids = tok(p_text, add_special_tokens=True)["input_ids"]
            p_ids = _truncate_prompt(p_ids, self.cfg.max_prompt_len, self.cfg.truncation_side)
            teacher_prompts.append(p_ids)

        return _collate_teacher_prompts_and_responses(batch, rt, tok, teacher_prompts)


@TEACHER_INPUTS.register("reason_first", source="Self-Distill/OPSD-main/opsd_trainer.py:1308-1362")
class ReasonFirst(TeacherInputBuilder):
    """Teacher writes an analysis of the reference solution first and uses it as context (OPSD reason_first)."""

    provides = ("teacher.input_ids", "teacher.attention_mask", "teacher.labels", "teacher.prompt_len")

    class Config(BaseConfig):
        reasoning_template: str
        transition_template: str
        fields: list[str] = ["solution"]
        sampling: dict = {}
        use_chat_template: bool = True
        chat_template_kwargs: dict = {}
        max_prompt_len: int = 4096
        truncation_side: str = "left"
        tokenizer: str = "student"

    def build(self, batch: Batch, rt: Any) -> Batch:
        if getattr(rt, "rollout", None) is None:
            raise ValueError("reason_first teacher input requires 'rollout:' to be configured in runtime")

        tok = (
            rt.student_tokenizer
            if self.cfg.tokenizer == "student"
            else (rt.teacher_tokenizer or rt.student_tokenizer)
        )
        batch_size = batch.batch_size
        r_prompt_ids_list: list[list[int]] = []
        meta_list: list[dict[str, Any]] = []

        for b in range(batch_size):
            meta_b = {k.removeprefix("meta."): v[b] for k, v in batch.meta.items() if k.startswith("meta.")}
            for f in self.cfg.fields:
                if f not in meta_b:
                    raise KeyError(
                        f"ReasonFirst: required meta field {f!r} missing. Available meta: {sorted(meta_b)}"
                    )
            meta_list.append(meta_b)
            r_text_b = self.cfg.reasoning_template.format(**meta_b)
            r_p_text = render_prompt(tok, r_text_b, self.cfg.use_chat_template, **self.cfg.chat_template_kwargs)
            r_p_ids = tok(r_p_text, add_special_tokens=True)["input_ids"]
            r_p_ids = _truncate_prompt(r_p_ids, self.cfg.max_prompt_len, self.cfg.truncation_side)
            r_prompt_ids_list.append(r_p_ids)

        spec = SamplingSpec(**{**self.cfg.sampling, "n": 1})
        rolls = rt.rollout.generate(r_prompt_ids_list, spec)

        teacher_prompts: list[list[int]] = []
        for b in range(batch_size):
            meta_b = meta_list[b]
            try:
                t_text = self.cfg.transition_template.format(**meta_b)
            except Exception:
                t_text = self.cfg.transition_template
            t_ids = tok(t_text, add_special_tokens=False)["input_ids"]

            roll = rolls[b][0]
            resp_ids = list(roll.response_ids)
            if tok.eos_token_id is not None and resp_ids and resp_ids[-1] == tok.eos_token_id:
                resp_ids = resp_ids[:-1]

            body = resp_ids + t_ids
            prompt_prefix = r_prompt_ids_list[b]
            if len(prompt_prefix) + len(body) > self.cfg.max_prompt_len:
                avail = max(0, self.cfg.max_prompt_len - len(prompt_prefix))
                t_prompt = prompt_prefix + body[:avail]
            else:
                t_prompt = prompt_prefix + body
            teacher_prompts.append(t_prompt)

        return _collate_teacher_prompts_and_responses(batch, rt, tok, teacher_prompts)


@TEACHER_INPUTS.register("sdpo_reprompt",
                         source="Self-Distill/SDPO-main/verl/trainer/ppo/ray_trainer.py:611-797")
class SDPOReprompt(TeacherInputBuilder):
    """SDPO teacher prompts use the first successful peer response and/or verifier feedback."""
    requires = ("student.input_ids", "student.prompt_len", "student.attention_mask", "rollout.reward", "group.uid")
    provides = ("teacher.input_ids", "teacher.attention_mask", "teacher.labels", "teacher.prompt_len", "rollout.has_signal")
    class Config(BaseConfig):
        reprompt_template: str
        solution_template: str = ""
        feedback_template: str = ""
        success_threshold: float = 0.5
        dont_reprompt_on_self_success: bool = True
        include_feedback: bool = True
        max_reprompt_len: int = Field(default=10240, gt=0)
        feedback_only_without_solution: bool = False
        remove_thinking_from_demonstration: bool = False
        truncation_side: str = "left"
        chat_template_kwargs: dict = {}
        write_loss_mask: bool = True

    @classmethod
    def provided_keys(cls, cfg):
        return cls.provides + (("seq.loss_mask",) if cfg.write_loss_mask else ())

    def _select(self, batch):
        uids, texts = batch.meta["group.uid"], batch.meta["meta.response_text"]
        success: dict[str, list[int]] = {uid: [] for uid in uids}
        for i, uid in enumerate(uids):
            if float(batch["rollout.reward"][i]) >= self.cfg.success_threshold:
                success[uid].append(i)
        solutions, feedbacks = [], []
        available = batch.meta.get("meta.feedback", [""] * len(uids))
        for i, uid in enumerate(uids):
            peers = [j for j in success[uid] if j != i or not self.cfg.dont_reprompt_on_self_success]
            solution = texts[peers[0]] if peers else None
            if solution is not None and self.cfg.remove_thinking_from_demonstration:
                solution = re.sub(r"<think>.*?</think>\s*", "", solution, flags=re.DOTALL)
            feedback = available[i] if self.cfg.include_feedback else ""
            if self.cfg.feedback_only_without_solution and solution is not None:
                feedback = ""
            solutions.append(solution)
            feedbacks.append(feedback)
        return success, solutions, feedbacks

    def build(self, batch: Batch, rt: Any) -> Batch:
        batch = batch.copy()
        tok = rt.student_tokenizer
        success, solutions, feedbacks = self._select(batch)
        prompts, signals = [], []
        for i, (solution, feedback) in enumerate(zip(solutions, feedbacks)):
            messages = batch.meta.get("meta.messages", [None] * batch.batch_size)[i]
            raw_prompts = batch.meta.get("meta.raw_prompt") or batch.meta.get("meta.prompt_text") or []
            prompt = messages[-1]["content"] if messages else raw_prompts[i]
            has_signal = solution is not None or bool(feedback)
            if has_signal:
                prompt = self.cfg.reprompt_template.format(
                    prompt=prompt,
                    solution=self.cfg.solution_template.format(successful_previous_attempt=solution) if solution is not None else "",
                    feedback=self.cfg.feedback_template.format(feedback_raw=feedback) if feedback else "")
            system = [m for m in messages[:-1] if m["role"] == "system"] if messages else []
            text = render_prompt(tok, system + [{"role": "user", "content": prompt}], True,
                                 **self.cfg.chat_template_kwargs)
            ids = tok(text, add_special_tokens=not bool(tok.chat_template))["input_ids"]
            prompts.append(_truncate_prompt(ids, self.cfg.max_reprompt_len, self.cfg.truncation_side))
            signals.append(float(has_signal))
        batch = _collate_teacher_prompts_and_responses(batch, rt, tok, prompts)
        signal = torch.tensor(signals, device=batch["student.input_ids"].device)
        batch["rollout.has_signal"] = signal
        if self.cfg.write_loss_mask:
            batch["seq.loss_mask"] = signal[:, None].expand_as(batch["student.labels"]).clone() * batch.get("seq.loss_mask", 1.)
        rt.state.log("sdpo/success_group_fraction", sum(bool(v) for v in success.values()) / len(success))
        rt.state.log("sdpo/success_sample_fraction", sum(s is not None for s in solutions) / batch.batch_size)
        rt.state.log("sdpo/feedback_used_fraction", sum(bool(f) for f in feedbacks) / batch.batch_size)
        rt.state.log("sdpo/reprompt_sample_fraction", signal.mean())
        return batch


@TEACHER_INPUTS.register("retokenize", source="IMPACT/code/sra_span_utils.py:283-314 (+ DESIGN §5.6)")
class Retokenize(TeacherInputBuilder):
    """Cross-tokenizer: re-tokenize the student text (incl. on-policy rollouts) for the teacher, then run aligners.

    Per row: ``prompt_text = decode(student prompt)``, ``resp_text = decode(response, skip_special_tokens=True)``;
    the teacher ids come from ``tokenize_prompt_response`` (+ offsets), the student offsets from ``ids_offsets``
    (rollout ids need not round-trip through text). A row whose retokenization or alignment fails keeps its place
    with ``seq.loss_mask = 0`` (batch size never changes). ``student.*`` of the batch is left untouched.
    """
    requires = ("student.input_ids", "student.prompt_len")
    provides = ("teacher.input_ids", "teacher.attention_mask", "teacher.labels", "teacher.prompt_len",
                "teacher.offsets", "student.offsets")

    class Config(BaseConfig):
        aligners: list[dict | str] = []     # run online after re-tokenizing, e.g. [{type: exact_token}]
        only_if_missing: bool = True        # batch already has teacher.* (off-policy): keep it

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        keys = tuple(cls.provides)
        for spec in (cfg.aligners if cfg is not None else []):
            meta, acfg = ALIGNERS.validate(spec)
            keys += tuple(meta.obj.provided_keys(acfg))
        return keys

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        self.aligners = [ALIGNERS.build(spec) for spec in self.cfg.aligners]
        for a in self.aligners:
            a.setup(rt)

    def _row_sample(self, ids: list[int], plen: int, rt: Any) -> dict[str, Any] | None:
        """Retokenized ``student.*``/``teacher.*`` offsets sample of one row, or None on failure."""
        from lmkd.data.spans import ids_offsets, prompt_response_offsets
        from lmkd.data.transforms.tokenize import tokenize_prompt_response

        stu, tea = rt.student_tokenizer, rt.teacher_tokenizer
        p_text = stu.decode(ids[:plen], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        bos_len = 0
        if stu.bos_token_id is not None and ids[0] == stu.bos_token_id and p_text.startswith(stu.bos_token):
            bos_len = len(stu.bos_token)         # the teacher adds its own BOS
            p_text = p_text[bos_len:]
        r_text = stu.decode(ids[plen:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        res = tokenize_prompt_response(tea, p_text, r_text, rt.config.data.max_length, True, "right")
        if res is None:
            return None
        t_ids, t_plen = res
        t_span = prompt_response_offsets(tea, p_text, r_text, t_ids, t_plen, add_eos=True)
        if not t_span.ok or t_span.offsets is None:
            return None
        p_off = [(max(s - bos_len, 0), max(e - bos_len, 0)) for s, e in ids_offsets(stu, ids[:plen])]
        r_off = ids_offsets(stu, ids[plen:], len(p_text), skip_special_tokens=True)
        return {"student.input_ids": ids, "student.prompt_len": plen, "teacher.input_ids": t_ids,
                "teacher.attention_mask": [1] * len(t_ids), "teacher.labels": make_labels(t_ids, t_plen).tolist(),
                "teacher.prompt_len": t_plen, "student.offsets": [list(o) for o in p_off + r_off],
                "teacher.offsets": [list(o) for o in t_span.offsets], "meta.prompt_text": p_text,
                "meta.response_text": r_text, "meta.teacher_prompt_text": p_text}

    def _fallback_sample(self, ids: list[int], plen: int, rt: Any) -> dict[str, Any]:
        """Placeholder teacher row (masked out of the loss) for a row that failed."""
        eos = rt.teacher_tokenizer.eos_token_id or 0
        return {"student.input_ids": ids, "student.prompt_len": plen, "teacher.input_ids": [eos, eos],
                "teacher.attention_mask": [1, 1], "teacher.labels": [-100, -100], "teacher.prompt_len": 1,
                "student.offsets": [[0, 0]] * len(ids), "teacher.offsets": [[0, 0], [0, 0]]}

    def _align(self, sample: dict[str, Any], rt: Any) -> bool:
        """Run the aligners into ``sample``; False (and ``empty`` values) if one raises ``ValueError``."""
        for a in self.aligners:
            try:
                sample.update(a(sample, rt.data_ctx))
            except ValueError:
                for b in self.aligners:
                    sample.update(b.empty(sample))
                return False
        return True

    def build(self, batch: Batch, rt: Any) -> Batch:
        if self.cfg.only_if_missing and "teacher.input_ids" in batch:
            return batch
        batch.require(*self.requires)
        if rt.teacher_tokenizer is None:
            raise ValueError("retokenize needs models.teacher (a teacher tokenizer)")
        mask = batch.get("student.attention_mask")
        samples, failed = [], []
        for b in range(batch.batch_size):
            n = int(mask[b].sum()) if mask is not None else batch["student.input_ids"].shape[1]
            ids = batch["student.input_ids"][b, :n].tolist()
            plen = int(batch["student.prompt_len"][b])
            row = self._row_sample(ids, plen, rt)
            ok = row is not None and self._align(row, rt)
            if row is None:
                row = self._fallback_sample(ids, plen, rt)
                self._align(row, rt)
            samples.append({k: v for k, v in row.items() if not k.startswith("meta.")})
            if not ok:
                failed.append(b)
        self._merge(batch, rt.collator(samples), failed)
        return batch

    @staticmethod
    def _merge(batch: Batch, new: Batch, failed: list[int]) -> None:
        """Copy the keys built by ``rt.collator`` into ``batch`` (student padding length of ``batch`` wins)."""
        from lmkd.core.batch import FIELDS

        t_len = batch["student.input_ids"].shape[1]
        device = batch["student.input_ids"].device
        for key in new:
            val = new[key]
            if key.startswith("student.") and key != "student.offsets" and key in batch:
                continue
            spec = FIELDS.get(key)
            if spec.kind in ("token", "token_k") and spec.side == "student" and val.shape[1] != t_len:
                fixed = torch.full((val.shape[0], t_len, *val.shape[2:]), spec.pad_value, dtype=val.dtype)
                n = min(t_len, val.shape[1])
                fixed[:, :n] = val[:, :n]
                val = fixed
            batch[key] = [v.to(device) for v in val] if isinstance(val, list) else val.to(device)
        if failed:
            loss_mask = batch["seq.loss_mask"].clone() if "seq.loss_mask" in batch else \
                (batch["student.labels"] != -100).float()
            loss_mask[failed] = 0.0
            batch["seq.loss_mask"] = loss_mask
