"""Tokenization transforms (reference implementations to imitate)."""

from __future__ import annotations

from typing import Any, Literal

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.labels import make_labels
from lmkd.data.transforms.base import Transform


def render_prompt(
    tokenizer: Any,
    prompt: Any,
    use_chat_template: bool,
    system_prompt: str | None = None,
    **tmpl_kwargs,
) -> str:
    """prompt may be a string or a list of chat messages."""
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        if isinstance(prompt, list):
            msgs = list(prompt)
            if system_prompt and (not msgs or msgs[0].get("role") != "system"):
                msgs = [{"role": "system", "content": system_prompt}] + msgs
        else:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": str(prompt)})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tmpl_kwargs)
    if isinstance(prompt, list):
        text = "\n".join(m["content"] for m in prompt)
        return f"{system_prompt}\n\n{text}" if system_prompt else text
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}"
    return prompt


def tokenize_prompt_response(
    tokenizer: Any,
    prompt_text: str,
    response_text: str,
    max_length: int,
    add_eos: bool,
    truncation: str,
    prompt_add_special_tokens: bool = True,
) -> tuple[list[int], int] | None:
    """Tokenize prompt and response SEPARATELY and concatenate ids (no BPE merge across the
    boundary -> prompt_len is exact). Returns (ids, prompt_len) or None if dropped.
    ``prompt_add_special_tokens=False`` when the chat template already contains BOS (SGL does this)."""
    p_ids = tokenizer(prompt_text, add_special_tokens=prompt_add_special_tokens)["input_ids"]
    r_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    if add_eos and tokenizer.eos_token_id is not None:
        r_ids = r_ids + [tokenizer.eos_token_id]
    ids = p_ids + r_ids
    if len(ids) > max_length:
        if truncation == "drop":
            return None
        if truncation == "right":
            ids = ids[:max_length]
        elif truncation == "left_prompt":
            cut = len(ids) - max_length
            if cut >= len(p_ids):
                return None
            p_ids = p_ids[cut:]
            ids = p_ids + r_ids
        else:
            raise ValueError(f"unknown truncation {truncation!r}")
    if len(p_ids) >= len(ids):
        return None  # no response token left
    return ids, len(p_ids)


@TRANSFORMS.register("tokenize_sft")
class TokenizeSFT(Transform):
    """{prompt, response} -> student.(input_ids, attention_mask, labels, prompt_len)
    (+ teacher.* with the teacher tokenizer if ``teacher: true``)."""

    provides = ("student.input_ids", "student.attention_mask", "student.labels", "student.prompt_len")

    class Config(BaseConfig):
        prompt_field: str = "prompt"
        response_field: str = "response"
        use_chat_template: bool = True
        add_eos: bool = True
        prompt_add_special_tokens: bool = True  # False if the rendered template already has BOS (K14)
        truncation: str = "drop"  # drop | right | left_prompt
        teacher: bool = False  # also tokenize for teacher (cross-tokenizer)
        teacher_prompt_text: Literal["own_template", "same"] = "own_template"  # same: teacher reads student's p_text
        keep_text: bool = True  # store meta.prompt_text / meta.response_text
        chat_template_kwargs: dict = {}
        system_prompt: str | None = None

    def _one(self, tok, prompt, response, ctx, side, out, p_text: str | None = None) -> bool:
        if p_text is None:
            p_text = render_prompt(
                tok,
                prompt,
                self.cfg.use_chat_template,
                system_prompt=self.cfg.system_prompt,
                **self.cfg.chat_template_kwargs,
            )
        res = tokenize_prompt_response(
            tok,
            p_text,
            response,
            ctx.max_length,
            self.cfg.add_eos,
            self.cfg.truncation,
            self.cfg.prompt_add_special_tokens,
        )
        if res is None:
            return False
        ids, plen = res
        out[f"{side}.input_ids"] = ids
        out[f"{side}.attention_mask"] = [1] * len(ids)
        out[f"{side}.labels"] = make_labels(ids, plen).tolist()
        out[f"{side}.prompt_len"] = plen
        if self.cfg.keep_text and side == "student":
            out["meta.prompt_text"] = p_text
            out["meta.response_text"] = response
        if side == "teacher":
            out["meta.teacher_prompt_text"] = p_text
        return True

    def _student_prompt_text(self, prompt: Any, ctx: DataContext) -> str:
        return render_prompt(
            ctx.student_tokenizer,
            prompt,
            self.cfg.use_chat_template,
            system_prompt=self.cfg.system_prompt,
            **self.cfg.chat_template_kwargs,
        )

    def __call__(self, sample: dict, ctx: DataContext) -> dict | None:
        prompt = sample.get(self.cfg.prompt_field)
        if prompt is None:
            prompt = sample.get("prompt") or sample.get("instruction") or sample.get("problem") or sample.get("question") or sample.get("Question")

        response = sample.get(self.cfg.response_field)
        if response is None:
            response = (
                sample.get("response")
                or sample.get("output")
                or sample.get("solution")
                or sample.get("target")
                or sample.get("ground_truth_solution")
                or sample.get("deepseek_solution")
                or sample.get("answer")
                or sample.get("Answer")
            )

        if (prompt is None or response is None) and "conversations" in sample:
            convs = sample["conversations"]
            if isinstance(convs, list) and convs:
                if prompt is None:
                    prompt = next(
                        (
                            c.get("value") or c.get("content")
                            for c in convs
                            if c.get("from") in ("user", "human") or c.get("role") in ("user", "human")
                        ),
                        None,
                    )
                if response is None:
                    response = next(
                        (
                            c.get("value") or c.get("content")
                            for c in convs
                            if c.get("from") in ("gpt", "assistant") or c.get("role") in ("gpt", "assistant")
                        ),
                        None,
                    )

        if (prompt is None or response is None) and "messages" in sample:
            msgs = sample["messages"]
            if isinstance(msgs, list) and msgs:
                if prompt is None:
                    prompt = next(
                        (
                            m.get("content") or m.get("value")
                            for m in msgs
                            if m.get("role") in ("user", "human") or m.get("from") in ("user", "human")
                        ),
                        None,
                    )
                if response is None:
                    response = next(
                        (
                            m.get("content") or m.get("value")
                            for m in msgs
                            if m.get("role") in ("assistant", "gpt") or m.get("from") in ("assistant", "gpt")
                        ),
                        None,
                    )

        if prompt is None:
            raise KeyError(f"Neither '{self.cfg.prompt_field}', 'prompt', nor 'instruction' found in sample: {list(sample.keys())}")
        if response is None:
            raise KeyError(f"Neither '{self.cfg.response_field}', 'response', nor 'output' found in sample: {list(sample.keys())}")
        if isinstance(response, list):
            response = response[0] if response else ""

        out = {k: v for k, v in sample.items() if k.startswith("meta.")}
        if not self._one(ctx.student_tokenizer, prompt, response, ctx, "student", out):
            return None
        if self.cfg.teacher:
            if ctx.teacher_tokenizer is None:
                raise ValueError("tokenize_sft(teacher=true) needs a teacher tokenizer")
            same = self.cfg.teacher_prompt_text == "same"
            p_text = self._student_prompt_text(prompt, ctx) if same else None
            if not self._one(ctx.teacher_tokenizer, prompt, response, ctx, "teacher", out, p_text):
                return None
        return out

    @classmethod
    def provided_keys(cls, cfg) -> tuple[str, ...]:
        keys = tuple(cls.provides)
        if cfg is not None and cfg.teacher:
            keys += ("teacher.input_ids", "teacher.attention_mask", "teacher.labels", "teacher.prompt_len")
        return keys


@TRANSFORMS.register("tokenize_text")
class TokenizeText(Transform):
    """{text} -> student.(input_ids, attention_mask) without labels (encoders / embedding KD).
    Truncates to data.max_length (embedding inputs are truncated, not dropped)."""

    provides = ("student.input_ids", "student.attention_mask")

    class Config(BaseConfig):
        text_field: str = "text"
        add_special_tokens: bool = True
        keep_text: bool = False
        teacher: bool = False

    def __call__(self, sample: dict, ctx: DataContext) -> dict | None:
        text = sample[self.cfg.text_field]
        ids = ctx.student_tokenizer(
            text, add_special_tokens=self.cfg.add_special_tokens, truncation=True, max_length=ctx.max_length
        )["input_ids"]
        if not ids:
            return None
        out = {k: v for k, v in sample.items() if k.startswith("meta.")}
        out["student.input_ids"] = ids
        out["student.attention_mask"] = [1] * len(ids)
        if self.cfg.teacher:
            if ctx.teacher_tokenizer is None:
                raise ValueError("tokenize_text(teacher=true) needs a teacher tokenizer")
            t_ids = ctx.teacher_tokenizer(
                text, add_special_tokens=self.cfg.add_special_tokens, truncation=True, max_length=ctx.max_length
            )["input_ids"]
            if not t_ids:
                return None
            out["teacher.input_ids"] = t_ids
            out["teacher.attention_mask"] = [1] * len(t_ids)
        if self.cfg.keep_text:
            out["meta.text"] = text
        return out

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        keys = tuple(cls.provides)
        if cfg is not None and getattr(cfg, "teacher", False):
            keys += ("teacher.input_ids", "teacher.attention_mask")
        return keys


@TRANSFORMS.register("tokenize_prompt")
class TokenizePrompt(Transform):
    """Prompt-only rows for on-policy/RL: student.input_ids = prompt; labels all -100."""

    provides = ("student.input_ids", "student.attention_mask", "student.labels", "student.prompt_len")

    class Config(BaseConfig):
        template: str = "{problem}"
        use_chat_template: bool = True
        chat_template_kwargs: dict = {}
        keep_fields: list[str] = ["problem", "solution", "answer", "target"]
        max_prompt_len: int | None = None

    def __call__(self, sample: dict, ctx: DataContext) -> dict | None:
        sample_dict = dict(sample)
        if "problem" not in sample_dict and "question" in sample_dict:
            sample_dict["problem"] = sample_dict["question"]
        if "problem" not in sample_dict and "Question" in sample_dict:
            sample_dict["problem"] = sample_dict["Question"]
        if "solution" not in sample_dict and "answer" in sample_dict:
            sample_dict["solution"] = sample_dict["answer"]
        if "solution" not in sample_dict and "Answer" in sample_dict:
            sample_dict["solution"] = sample_dict["Answer"]
        if "solution" not in sample_dict and "ground_truth_solution" in sample_dict:
            sample_dict["solution"] = sample_dict["ground_truth_solution"]
        if "solution" not in sample_dict and "deepseek_solution" in sample_dict:
            sample_dict["solution"] = sample_dict["deepseek_solution"]
        if "solution" not in sample_dict and "chosen" in sample_dict:
            chosen = sample_dict["chosen"]
            if isinstance(chosen, list) and chosen:
                sample_dict["solution"] = str(chosen[-1].get("content", ""))
            elif isinstance(chosen, str):
                sample_dict["solution"] = chosen

        if "problem" not in sample_dict and "conversations" in sample_dict:
            convs = sample_dict["conversations"]
            if isinstance(convs, list) and convs:
                user_msg = next(
                    (
                        c.get("value") or c.get("content")
                        for c in convs
                        if c.get("from") in ("user", "human") or c.get("role") in ("user", "human")
                    ),
                    None,
                )
                asst_msg = next(
                    (
                        c.get("value") or c.get("content")
                        for c in convs
                        if c.get("from") in ("gpt", "assistant") or c.get("role") in ("gpt", "assistant")
                    ),
                    None,
                )
                if user_msg:
                    sample_dict["problem"] = user_msg
                if asst_msg and "solution" not in sample_dict:
                    sample_dict["solution"] = asst_msg

        if "problem" not in sample_dict and "messages" in sample_dict:
            msgs = sample_dict["messages"]
            if isinstance(msgs, list) and msgs:
                user_msg = next(
                    (
                        m.get("content") or m.get("value")
                        for m in msgs
                        if m.get("role") in ("user", "human") or m.get("from") in ("user", "human")
                    ),
                    None,
                )
                asst_msg = next(
                    (
                        m.get("content") or m.get("value")
                        for m in msgs
                        if m.get("role") in ("assistant", "gpt") or m.get("from") in ("assistant", "gpt")
                    ),
                    None,
                )
                if user_msg:
                    sample_dict["problem"] = user_msg
                if asst_msg and "solution" not in sample_dict:
                    sample_dict["solution"] = asst_msg

        try:
            prompt_str = self.cfg.template.format(**sample_dict)
        except KeyError as err:
            raise KeyError(f"Template references field {err} missing from sample: {sorted(sample_dict)}") from err

        tok = ctx.student_tokenizer
        p_text = render_prompt(tok, prompt_str, self.cfg.use_chat_template, **self.cfg.chat_template_kwargs)
        ids = tok(p_text, add_special_tokens=True)["input_ids"]

        if self.cfg.max_prompt_len is not None and len(ids) > self.cfg.max_prompt_len:
            return None
        if ctx.max_length is not None and len(ids) > ctx.max_length:
            return None
        if not ids:
            return None

        out = {k: v for k, v in sample_dict.items() if k.startswith("meta.")}
        for f in self.cfg.keep_fields:
            if f in sample_dict:
                out[f"meta.{f}"] = sample_dict[f]
        out["meta.raw_prompt"] = prompt_str
        out["meta.prompt_text"] = p_text

        plen = len(ids)
        out["student.input_ids"] = ids
        out["student.attention_mask"] = [1] * plen
        out["student.labels"] = [-100] * plen
        out["student.prompt_len"] = plen
        return out
