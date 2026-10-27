"""TeacherProvider: who the teacher is (DESIGN §5.5)."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, ClassVar

import torch

from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import TEACHERS
from lmkd.core.state import TrainState
from lmkd.models.forward import ForwardRunner, SideOutput, SideRequest, side_inputs
from lmkd.models.logits_view import LogitsView
from lmkd.rollout.base import ScoreSpec
from lmkd.teachers.signals import TeacherOutput, TeacherSignalSpec, gather_positions, response_shift_index


class TeacherProvider(Plugin):
    """Who the teacher is (axis C): turns a batch and a signal spec into a :class:`TeacherOutput`."""
    capabilities: ClassVar[frozenset[str]] = frozenset()   # signal kinds it can produce

    def setup(self, rt: Any) -> None:
        self.rt = rt

    def signals(self, batch: Batch, spec: TeacherSignalSpec | None, req: SideRequest | None,
                student: SideOutput | None) -> TeacherOutput | None:
        raise NotImplementedError

    def on_optimizer_step(self, state: TrainState) -> None:
        return None

    def check(self, spec: TeacherSignalSpec | None, req: SideRequest | None) -> None:
        if spec is not None and spec.kind != "none" and spec.kind not in self.capabilities:
            raise ValueError(
                f"teacher provider '{self.registry_name}' cannot produce signal '{spec.kind}' "
                f"(capabilities: {sorted(self.capabilities)})"
            )


@TEACHERS.register("none")
class NoTeacher(TeacherProvider):
    """No teacher (SFT, embedding SimCSE, ...)."""
    class Config(BaseConfig):
        pass

    def signals(self, batch, spec, req, student):
        return None


def _fill_from_view(out: TeacherOutput, batch: Batch, spec: TeacherSignalSpec | None, side: str,
                    student: SideOutput | None) -> None:
    """Derive sparse signals from an online LogitsView."""
    if spec is None or out.logits is None:
        return
    view = out.logits.with_temperature(spec.temperature) if spec.temperature != 1.0 else out.logits
    if spec.kind == "topk":
        if spec.support == "student":
            if student is None or student.logits is None:
                raise ValueError("support='student' needs student logits")
            ids, _ = student.logits.topk(spec.k)
            out.topk_ids, out.topk_logp = ids, view.gather(ids)
        else:
            out.topk_ids, out.topk_logp = view.topk(spec.k)
        if spec.tail == "bucket":
            mass = out.topk_logp.exp().sum(-1).clamp(max=1 - 1e-6)
            out.tail_logp = torch.log1p(-mass)
    elif spec.kind in ("sampled_logp", "seq_logp"):
        labels = batch[f"{side}.labels"]
        out.token_logp = view.target_logp(labels)
        if spec.kind == "seq_logp":
            out.seq_logp = out.token_logp.sum(-1)


@TEACHERS.register("hf_frozen")
class HFFrozenTeacher(TeacherProvider):
    """Separate frozen HF model (rt.teacher_model), run under no_grad on ``teacher.*`` inputs
    (falls back to ``student.*`` when the teacher input builder is ``same``).

    ``position_map``:
      * ``identity``       teacher positions == student positions (same input, or cross-tokenizer
                           criteria that align via ``align.*`` themselves);
      * ``response_shift`` same tokenizer, teacher prompt differs (privileged context) but response
                           tokens are identical: the teacher LogitsView (and every sparse signal
                           derived from it) is re-indexed onto STUDENT positions (K8). ``side``
                           (hidden states) stays in teacher positions.
    """

    capabilities = frozenset({"full", "topk", "sampled_logp", "seq_logp", "hidden"})

    class Config(BaseConfig):
        position_map: str = "identity"   # identity | response_shift

    def signals(self, batch, spec, req, student):
        model = self.rt.teacher_model
        if model is None:
            raise ValueError("hf_frozen teacher: models.teacher is not configured")
        side = "teacher" if "teacher.input_ids" in batch else "student"
        req = req or SideRequest(logits="view")
        if spec is not None and spec.kind in ("full", "topk", "sampled_logp", "seq_logp") and req.logits == "none":
            req = SideRequest(**{**req.__dict__, "logits": "view"})
        outs = ForwardRunner.run(model, self.rt.teacher_adapter, side_inputs(batch, side), req, grad=False)
        if not outs:
            return TeacherOutput()
        so = outs[0]
        view = so.logits
        label_side = side
        if self.cfg.position_map == "response_shift" and side == "teacher" and view is not None:
            idx, _ = response_shift_index(batch, so.last_hidden.shape[1])
            view = LogitsView(gather_positions(so.last_hidden, idx), self.rt.teacher_adapter.lm_head(model))
            label_side = "student"   # signals are now indexed like the student
        out = TeacherOutput(logits=view, side=so)
        _fill_from_view(out, batch, spec, label_side, student)
        return out


@TEACHERS.register("self")
class SelfTeacher(TeacherProvider):
    """The student itself (with a WeightPolicy) on privileged teacher inputs (OPSD/SDPO/RLSD/PBSD)."""

    capabilities = frozenset({"full", "topk", "sampled_logp", "seq_logp", "hidden"})

    class Config(BaseConfig):
        weight_policy: dict | str = "lora_disabled"
        position_map: str = "response_shift"   # response_shift | identity
        eval_mode: bool = True                 # disable dropout during teacher forward

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        from lmkd.core.registry import WEIGHT_POLICIES

        self.policy = WEIGHT_POLICIES.build(self.cfg.weight_policy)
        self.policy.setup(rt)

    def signals(
        self,
        batch: Batch,
        spec: TeacherSignalSpec | None,
        req: SideRequest | None,
        student: SideOutput | None,
    ) -> TeacherOutput | None:
        side = "teacher" if "teacher.input_ids" in batch else "student"
        req = req or SideRequest(logits="view")
        if spec is not None and spec.kind in ("full", "topk", "sampled_logp", "seq_logp") and req.logits == "none":
            req = SideRequest(**{**req.__dict__, "logits": "view"})

        from lmkd.models.adapters import get_base_head
        from lmkd.teachers.weight_policies import EMA, LoraDisabled, TrustRegion
        if isinstance(self.policy, TrustRegion):
            return self._trust_region_signals(batch, spec, req, student, side)

        with self.policy.context(self.rt) as model:
            was = model.training
            if self.cfg.eval_mode:
                model.eval()
            try:
                # Run on raw model outside DDP wrapper without gradients
                outs = ForwardRunner.run(model, self.rt.student_adapter, side_inputs(batch, side), req, grad=False)
            finally:
                if self.cfg.eval_mode and was:
                    model.train()

        if not outs:
            return TeacherOutput()
        so = outs[0]
        signal_head = None
        if isinstance(self.policy, LoraDisabled):
            signal_head = get_base_head(self.rt.student_adapter.lm_head(self.rt.student_model))
        elif isinstance(self.policy, EMA) and self.policy.ema and req.logits != "none":
            signal_head = self.policy.get_head(self.rt)

        view = LogitsView(so.last_hidden, signal_head) if signal_head is not None else so.logits
        label_side = side

        if side == "teacher" and self.cfg.position_map == "response_shift":
            if req.hidden != "none":
                raise NotImplementedError("response_shift position_map does not support intermediate hidden states")
            if view is not None:
                idx, _ = response_shift_index(batch, so.last_hidden.shape[1])
                head = signal_head if signal_head is not None else self.rt.student_adapter.lm_head(model)
                view = LogitsView(gather_positions(so.last_hidden, idx), head)
                label_side = "student"

        out = TeacherOutput(logits=view, side=so)
        _fill_from_view(out, batch, spec, label_side, student)
        return out

    def _trust_side(self, model, batch, req, side):
        was = model.training
        if self.cfg.eval_mode:
            model.eval()
        try:
            so = ForwardRunner.run(model, self.rt.student_adapter, side_inputs(batch, side), req, grad=False)[0]
        finally:
            model.train(was)
        view = so.logits
        if side == "teacher" and self.cfg.position_map == "response_shift":
            if req.hidden != "none":
                raise NotImplementedError("response_shift does not support intermediate hidden states")
            if view is not None:
                idx, _ = response_shift_index(batch, so.last_hidden.shape[1])
                view = LogitsView(gather_positions(so.last_hidden, idx), self.rt.student_adapter.lm_head(model))
        return so, view

    def _trust_region_signals(self, batch, spec, req, student, side):
        import warnings

        head = self.rt.student_adapter.lm_head(self.rt.student_model)
        has_lora = head is not None and (hasattr(head, "base_layer") or hasattr(head, "original_module"))
        shared = self.policy.ref is None and not has_lora and not any(p.requires_grad for p in head.parameters())
        ref_logits = None

        with torch.no_grad(), self.policy.context(self.rt) as ref:
            ref_side, ref_view = self._trust_side(ref, batch, req, side)
            if not shared and ref_view is not None:
                ref_logits = ref_view.positional()

        with torch.no_grad():
            live_side, live_view = self._trust_side(self.rt.student_model, batch, req, side)
            view = None
            if ref_view is not None and live_view is not None:
                if shared:
                    if getattr(head, "bias", None) is not None:
                        raise ValueError("trust_region hidden interpolation requires a bias-free lm_head")
                    view = LogitsView(torch.lerp(ref_view.hidden, live_view.hidden, self.policy.cfg.mix), head)
                else:
                    if not self.policy.warned_full:
                        warnings.warn("trust_region materializes logits because the reference and live heads differ", stacklevel=2)
                        self.policy.warned_full = True
                    assert ref_logits is not None
                    mixed = torch.lerp(ref_logits, live_view.positional(), self.policy.cfg.mix)
                    view = LogitsView(logits=mixed)
        label_side = "student" if side == "teacher" and self.cfg.position_map == "response_shift" else side
        out = TeacherOutput(logits=view, side=live_side)
        _fill_from_view(out, batch, spec, label_side, student)
        return out

    def on_optimizer_step(self, state: TrainState) -> None:
        self.policy.on_optimizer_step(self.rt)

    def state_dict(self) -> dict[str, Any]:
        return self.policy.state_dict() if hasattr(self.policy, "state_dict") else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if hasattr(self.policy, "load_state_dict"):
            self.policy.load_state_dict(state)


@TEACHERS.register("cache", source="integrations/distillation_data.py:77-123,587-631")
class CacheTeacher(TeacherProvider):
    """Read target-aligned teacher top-k or log-probabilities from batch columns.

    Source: human_alignment/integrations/distillation_data.py:77-123,587-631.
    """

    capabilities = frozenset({"topk", "sampled_logp", "seq_logp"})

    class Config(BaseConfig):
        kind: str = "topk"

    @classmethod
    def required_keys(cls, cfg: Any) -> tuple[str, ...]:
        keys = {"topk": ("cache.topk_ids", "cache.topk_p", "cache.tail_p"),
                "token_logp": ("cache.token_logp",), "seq_logp": ("cache.seq_logp",)}
        if cfg.kind not in keys:
            raise ValueError(f"unknown cache kind {cfg.kind!r}")
        return keys[cfg.kind]

    def check(self, spec: TeacherSignalSpec | None, req: SideRequest | None) -> None:
        super().check(spec, req)
        if spec is None or spec.kind == "none":
            return
        allowed = {"topk": {"topk"}, "token_logp": {"sampled_logp", "seq_logp"},
                   "seq_logp": {"seq_logp"}}
        if self.cfg.kind not in allowed or spec.kind not in allowed[self.cfg.kind]:
            raise ValueError(f"cache kind {self.cfg.kind!r} cannot produce {spec.kind!r}")
        if spec.temperature != 1.0 or spec.support != "teacher":
            raise ValueError("offline cache needs temperature=1 and teacher support")
        if req is not None and (req.hidden != "none" or req.attentions != "none"):
            raise ValueError("offline cache cannot provide hidden or attention states")

    @staticmethod
    def _check_fingerprint(batch: Batch) -> None:
        fps = batch.meta.get("meta.cache_tokenize_fp")
        if fps is None:
            return
        from lmkd.data.transforms.listwise import tokenize_fingerprint

        lengths = batch["student.attention_mask"].sum(-1).tolist()
        ids = batch["student.input_ids"].tolist()
        labels = batch["student.labels"].tolist()
        for i, (fp, length) in enumerate(zip(fps, lengths)):
            if fp is None:
                continue
            actual = tokenize_fingerprint(ids[i][:length], labels[i][:length])
            if fp != actual:
                raise ValueError(f"cache tokenization fingerprint mismatch at row {i}")

    def signals(self, batch: Batch, spec: TeacherSignalSpec | None, req: SideRequest | None,
                student: SideOutput | None) -> TeacherOutput:
        self.check(spec, req)
        self._check_fingerprint(batch)
        if self.cfg.kind == "topk":
            ids = batch["cache.topk_ids"]
            probs = batch["cache.topk_p"]
            tail = batch["cache.tail_p"]
            if spec is not None and spec.k and spec.k > ids.shape[-1]:
                raise ValueError(f"cache has K={ids.shape[-1]} but signal asks for K={spec.k}")
            active = probs.sum((-1, -2)).gt(0)
            if active.any() and "student.labels" in batch:
                targets = batch["student.labels"].ne(-100)
                present = probs.sum(-1).gt(0)
                if not torch.equal(present[active], targets[active]):
                    raise ValueError("cache target count/positions differ from supervised labels")
            return TeacherOutput(topk_ids=ids, topk_logp=probs.clamp_min(1e-12).log(),
                                 tail_logp=tail.clamp_min(1e-12).log())
        if self.cfg.kind == "token_logp":
            token = batch["cache.token_logp"]
            mask = batch["student.labels"].ne(-100)
            return TeacherOutput(token_logp=token, seq_logp=(token * mask).sum(-1))
        return TeacherOutput(seq_logp=batch["cache.seq_logp"])


@TEACHERS.register("vllm_scorer", source="DESIGN §5.5 (vLLM prompt_logprobs)")
class VLLMScorerTeacher(TeacherProvider):
    """Large frozen teacher served by its own vLLM engine; returns top-k / log p(actual token) per position.

    Signals come from ``prompt_logprobs``: entry t is the distribution OF token t (target-aligned, entry 0
    is empty). Only the raw (T=1) top-k plus the actual token are available, so it cannot evaluate the
    teacher at arbitrary ids (``support="student"``), apply a temperature, or give hidden states.
    """

    capabilities = frozenset({"topk", "sampled_logp", "seq_logp"})

    class Config(BaseConfig):
        k: int = 20
        gpu_memory_utilization: float = 0.3
        tensor_parallel_size: int = 1
        max_model_len: int | None = None

    def setup(self, rt):
        from lmkd.rollout.vllm_colocate import make_llm, score_with_llm

        self.rt = rt
        spec = rt.config.models.teacher
        if spec is None:
            raise ValueError("vllm_scorer teacher: models.teacher is not configured")
        self._score = score_with_llm
        self.llm = make_llm(spec.path, spec.tokenizer or spec.path, world_size=rt.engine.world_size,
                            rank=rt.engine.rank, tensor_parallel_size=self.cfg.tensor_parallel_size,
                            dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[spec.dtype],
                            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                            max_model_len=self.cfg.max_model_len, enable_prefix_caching=True,
                            max_logprobs=max(20, self.cfg.k))
        self._same_vocab = (rt.teacher_tokenizer is None or rt.student_tokenizer is None
                            or rt.teacher_tokenizer.get_vocab() == rt.student_tokenizer.get_vocab())

    def check(self, spec, req):
        super().check(spec, req)
        why = ("vLLM prompt_logprobs only returns the raw top-k plus the actual token; it cannot be "
               "evaluated at arbitrary indices, rescaled or inspected")
        if req is not None and (req.hidden != "none" or req.attentions != "none"
                                or req.attention_last_row != "none" or req.input_embeds or req.pooled):
            raise ValueError(f"vllm_scorer cannot provide hidden/attention states: {why}")
        if spec is not None:
            if spec.support == "student":
                raise ValueError(f"vllm_scorer cannot score support='student': {why}")
            if spec.temperature != 1.0:
                raise ValueError(f"vllm_scorer needs teacher temperature 1.0 (got {spec.temperature}): {why}")
            if spec.kind == "topk" and spec.k > self.cfg.k:
                raise ValueError(f"signal k={spec.k} > vllm_scorer k={self.cfg.k}")

    def signals(self, batch, spec, req, student):
        self.check(spec, req)
        if spec is None or spec.kind == "none":
            return TeacherOutput()
        side = "teacher" if "teacher.input_ids" in batch else "student"
        if side == "teacher" and not self._same_vocab:
            raise NotImplementedError("vllm_scorer + cross-tokenizer teacher input: positions cannot be remapped")
        ids, am = batch[f"{side}.input_ids"], batch[f"{side}.attention_mask"]
        B, T = ids.shape
        lens = am.sum(-1).tolist()
        seqs = [ids[b, : int(lens[b])].tolist() for b in range(B)]
        k = spec.k if spec.kind == "topk" else 0
        pl = self._score(self.llm, seqs, ScoreSpec(top_k=k))
        out = TeacherOutput()
        token_logp = torch.zeros(B, T)
        if k:
            topk_ids = torch.zeros(B, T, k, dtype=torch.long)
            topk_logp = torch.full((B, T, k), float("-inf"))
        for b in range(B):
            for t in range(1, int(lens[b])):
                d = pl[b][t]
                token_logp[b, t] = d[seqs[b][t]]          # vLLM always includes the actual token
                if k:
                    top = sorted(d.items(), key=lambda kv: -kv[1])[:k]
                    topk_ids[b, t, : len(top)] = torch.tensor([i for i, _ in top])
                    topk_logp[b, t, : len(top)] = torch.tensor([v for _, v in top])
        dev = ids.device
        # like hf_frozen / LogitsView.target_logp: log p(label) on supervised positions, 0 elsewhere
        # (so seq_logp is the response log-likelihood, not prompt + response)
        token_logp = token_logp.to(dev) * (batch[f"{side}.labels"] != -100)
        out.token_logp = token_logp
        if spec.kind == "seq_logp":
            out.seq_logp = out.token_logp.sum(-1)
        if k:
            out.topk_ids = topk_ids.to(dev)
            out.topk_logp = torch.where(torch.isinf(topk_logp), torch.full_like(topk_logp, -1e4), topk_logp).to(dev)
            if spec.tail == "bucket":
                out.tail_logp = torch.log1p(-out.topk_logp.exp().sum(-1).clamp(max=1 - 1e-6))
        return out


@TEACHERS.register(
    "embedding_store",
    source="embedding-distill/TALAS-M-main/dataset.py:10-21",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
)
class EmbeddingStoreTeacher(TeacherProvider):
    """Offline embedding store teacher (TALAS).

    Loads sentence embeddings from a safetensors or pt file (or from a pipeline stage artifact)
    and looks up rows by `meta.id` (or by sequential position when `by_position=True`).
    """

    capabilities = frozenset({"embedding"})

    class Config(BaseConfig):
        node: str = "embed_teacher"
        path: str | None = None
        dtype: str = "fp32"
        by_position: bool = False

    def setup(self, rt: Any) -> None:
        self.rt = rt
        target_path, ids_path = self._locate(rt)
        if target_path is None or not target_path.exists():
            raise FileNotFoundError(f"Embeddings file {target_path} not found")
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.cfg.dtype, torch.float32)
        self.emb = self._read(target_path).to(dtype)
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):   # pinning is an optimization only
                self.emb = self.emb.pin_memory()
        self.index = self._read_index(ids_path)

    def _locate(self, rt: Any) -> tuple[Path | None, Path | None]:
        """(embeddings file, ids.json) from ``path`` (file or directory) or the pipeline node artifact."""
        def in_dir(d: Path) -> Path | None:
            for name in ("embeddings.safetensors", "embeddings.pt"):
                if (d / name).exists():
                    return d / name
            return None

        if self.cfg.path is not None:
            p = Path(self.cfg.path)
            if not p.is_dir():
                return p, p.parent / "ids.json"
            found = in_dir(p)
            if found is None:
                raise FileNotFoundError(f"No embeddings file found in directory {p}")
            return found, p / "ids.json"
        arts = getattr(rt, "extra", {}).get("pipeline_artifacts", {})
        if self.cfg.node not in arts:
            raise ValueError(f"Neither path nor pipeline node '{self.cfg.node}' found in runtime artifacts")
        art_dir = Path(arts[self.cfg.node].path)
        return in_dir(art_dir), art_dir / "ids.json"

    @staticmethod
    def _read(path: Path) -> torch.Tensor:
        data: Any
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            data = load_file(str(path))
        else:
            data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            for key in ("emb", "embedding"):
                if key in data:
                    data = data[key]
                    break
            else:
                raise KeyError(f"Expected 'emb' or 'embedding' in {path}")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for embeddings, got {type(data)}")
        return data

    def _read_index(self, ids_path: Path | None) -> dict[str, int]:
        """``meta.id`` -> row. Without ids.json rows are addressed by position (only if ``by_position``)."""
        if ids_path is not None and ids_path.exists():
            return {str(k): i for i, k in enumerate(json.loads(ids_path.read_text()))}
        if not self.cfg.by_position:
            raise ValueError(f"ids.json not found at {ids_path} and by_position is False")
        return {str(i): i for i in range(len(self.emb))}

    def signals(
        self,
        batch: Batch,
        spec: TeacherSignalSpec | None,
        req: SideRequest | None,
        student: SideOutput | None,
    ) -> TeacherOutput:
        ids = batch.meta.get("meta.id")
        if ids is None:
            raise KeyError("batch missing 'meta.id' in meta dict for embedding_store teacher")
        rows = []
        for i in ids:
            k = str(i)
            if k not in self.index:
                raise KeyError(f"ID {i!r} not found in embedding store index")
            rows.append(self.index[k])

        dev = (
            student.last_hidden.device
            if (student is not None and hasattr(student, "last_hidden"))
            else getattr(getattr(self.rt, "engine", None), "device", "cpu")
        )
        emb = self.emb[rows].to(dev)
        return TeacherOutput(embedding=emb)
