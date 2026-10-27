"""Online preference-based self-distillation using the frozen LoRA base."""
from types import SimpleNamespace

from lmkd.core.registry import PRODUCERS, REWARDS
from lmkd.data.producers import DatasetProducer
from lmkd.rollout.base import Rollout, prompts_from_batch, rollouts_to_samples, sampling_from_cfg
from lmkd.rollout.hf import HFRollout


@PRODUCERS.register("pbsd_online", source="Awesome-Human-AI-Alignment-main/src/human_alignment/integrations/self_distillation.py:248-303")
class PBSDOnline(DatasetProducer):
    """Generate teacher/student pairs and gate on teacher-correct/student-wrong.

    Teacher generation disables LoRA on the privileged prompt; both responses
    are scored/trained under the student's original prompt in pair order [T, S].
    Local HF generation preserves the base weights and resumes the prompt sampler.
    """
    provides = DatasetProducer.provides + ("group.index", "group.size", "group.valid", "group.uid", "pref.gate")

    class Config(DatasetProducer.Config):
        sampling: dict = {}
        verifier: dict | str = "math_verify"

    def setup(self, rt):
        if rt.engine.world_size != 1 or rt.engine.is_partitioned:
            raise ValueError("PBSD online currently requires local single-process HF training")
        if rt.config.train.batch_size % 2 or rt.config.train.micro_batch_size % 2:
            raise ValueError("PBSD batch and micro batch sizes must preserve two-response pairs")
        if rt.config.models.student.lora is None:
            raise ValueError("PBSD requires a LoRA student and a frozen adapter-disabled base")
        provider = rt.config.teacher.provider
        policy = provider.get("weight_policy") if isinstance(provider, dict) else None
        if isinstance(policy, dict):
            policy = policy.get("type")
        if policy != "lora_disabled" or not hasattr(rt.teacher, "policy"):
            raise ValueError("PBSD requires self teacher with weight_policy=lora_disabled")
        if not isinstance(rt.rollout, HFRollout):
            raise ValueError("PBSD online currently requires rollout.type=hf")
        # DatasetProducer counts prompt rows; each prompt produces two training rows.
        cfg = rt.config.model_copy(deep=True)
        cfg.train.batch_size //= 2
        proxy = SimpleNamespace(**{**vars(rt), "config": cfg})
        super().setup(proxy)
        self.rt = rt
        self.teacher_rollout = HFRollout(HFRollout.Config())
        self.teacher_rollout.setup(rt)
        self.verifier = REWARDS.build(self.cfg.verifier)
        self.verifier.setup(rt)
        self.spec = sampling_from_cfg({**self.cfg.sampling, "n": 1})

    def next(self, state):
        prompts = super().next(state)
        privileged = self.rt.teacher_input.build(prompts.copy(), self.rt)
        if "teacher.prompt_len" not in privileged:
            raise ValueError("PBSD needs a privileged_prompt teacher input")
        original_ids = prompts_from_batch(prompts)
        students = self.rt.rollout.generate(original_ids, self.spec)
        with self.rt.teacher.policy.context(self.rt) as model:
            proxy = SimpleNamespace(student_model=model, student_tokenizer=self.rt.student_tokenizer,
                                    engine=self.rt.engine)
            self.teacher_rollout.setup(proxy)
            teachers = self.teacher_rollout.generate(prompts_from_batch(privileged, "teacher"), self.spec)
        if len(students) != len(original_ids) or len(teachers) != len(original_ids):
            raise ValueError("PBSD generation returned an incorrect prompt count")
        if any(len(group) != 1 for group in students + teachers):
            raise ValueError("PBSD generation must return one response per prompt")
        return self.rt.collator(self._pairs(prompts, original_ids, teachers, students, state))

    def _pairs(self, prompts, ids, teachers, students, state):
        metas, texts, rolls = [], [], []
        for i, (teacher, student) in enumerate(zip(teachers, students)):
            meta = {k: v[i] for k, v in prompts.meta.items()}
            for generated in (teacher[0], student[0]):
                if not generated.response_ids:
                    raise ValueError("PBSD received an empty completion")
                rolls.append(Rollout(ids[i], generated.response_ids, text=generated.text))
                texts.append(generated.text)
                metas.append(meta)
        records = [{k.removeprefix("meta."): v for k, v in m.items()} for m in metas]
        results = self.verifier([m.get("prompt_text", "") for m in records], texts, records)
        if len(results) != len(rolls):
            raise ValueError("PBSD verifier must return one result per response")
        samples = rollouts_to_samples(rolls, copy_meta=metas)
        gates = []
        for i in range(0, len(samples), 2):
            gate = float(bool(results[i]["acc"]) and not bool(results[i + 1]["acc"]))
            gates.append(gate)
            for j in (0, 1):
                samples[i + j].update({"group.index": j, "group.size": 2, "group.valid": True,
                                       "group.uid": f"{metas[i].get('meta.id', i)}@{state.step}",
                                       "pref.gate": gate, "meta.response_text": texts[i + j]})
        state.log("pbsd/gate_fraction", sum(gates) / len(gates))
        return samples
