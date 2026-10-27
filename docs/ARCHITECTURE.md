# LM-KD in 15 minutes

Read this first. [DESIGN.md](DESIGN.md) is the full rationale (long), [CONTRIBUTING.md](CONTRIBUTING.md) the contract
for contributors, [README.md](../README.md) the commands. This page answers: *what happens in a training run, where
does each idea live, and where do I change X?*

For a runnable first experiment and current decorator/engine contracts, see
[NEW_METHOD_GUIDE.md](NEW_METHOD_GUIDE.md) and [examples/new_method.py](../examples/new_method.py).

## 1. One idea

Every distillation method is a point in a space of six independent choices. LM-KD has **one training loop** and
makes each choice a **plugin** you select in a YAML recipe (or from Python):

| Axis | Question | Plugin kind (registry) | Example |
|---|---|---|---|
| A data | how are samples made/transformed? | `transforms`, pipeline `stages` | tokenize, segment CoT, align tokenizers |
| B batches | where does each step's batch come from? | `producers` | dataset, replay, rollouts |
| C teacher | who is the teacher, what does it see? | `teachers`, `teacher_inputs` | frozen HF model, cached top-k, self |
| D objective | which loss, combined how? | `criterions`, `aggregators` | `nll`, `token_kd`, DSKD, DPO |
| E step | what is one optimizer step? | `step_strategies`, `grad_transforms` | standard, SAM, PCGrad |
| F optimizer | which params, optimizer, LR? | `optimizers`, `schedulers` | AdamW + Muon, warmup/cosine |

`lmkd list` prints every plugin with its status (`stable` works, `experimental` or planned).

## 2. What one training step does

```
Runtime.train_step()                                   src/lmkd/train/runtime.py
 ├─ callbacks.on_step_begin                            (loss-weight schedules, ...)
 ├─ batch = producer.next()                            axis B   -> Batch (dict "ns.key" -> tensor)
 ├─ algorithm.augment(batch)                           axis C   teacher inputs (+ RL signals)
 └─ step_strategy.run(batch)                           axis E   src/lmkd/train/step_strategies.py
      ├─ split into micro-batches; count supervised tokens ONCE for the whole step (all ranks)
      ├─ for each micro-batch:
      │    algorithm.loss(micro)                       src/lmkd/train/algorithms.py
      │      ├─ ForwardRunner: ONE student forward with exactly the features the criteria asked for
      │      ├─ teacher.signals(...)                   no_grad teacher forward / cache read
      │      └─ every criterion -> LossOutput (numerators)       axis D
      │    aggregator: divide by the global denominators, weight, sum
      │    engine.backward
      ├─ grad transforms (clip, ...) -> engine.step_optimizers -> schedulers      axis F
 ├─ update state, criterion.on_step_end, teacher.on_optimizer_step
 └─ callbacks.on_step_end                              logging, checkpoint, evaluation
```

`lmkd train` = build everything, then `Runtime.fit()` (= `begin`, `train_step` until done, `end`). A script can write
the same loop itself around `rt.train_step()` ([examples/custom_loop.py](../examples/custom_loop.py)).

## 3. The 12 concepts you need, and where they live

| Concept | One line | File |
|---|---|---|
| **Recipe** | the YAML; validated strictly (unknown key = error) | `core/config.py` |
| **Registry / plugin** | named, validated, documented components; `Config` = pydantic params | `core/registry.py`, `core/plugin.py` |
| **Batch + FieldSpec** | flat dict of namespaced tensors (`student.input_ids`, `seq.token_weights`); FieldSpec tells the collator how to pad | `core/batch.py`, `data/collator.py` |
| **Target-aligned convention** | position *t* of every per-token tensor describes token *t*; only `LogitsView` shifts logits | `models/logits_view.py` (DESIGN §4.2) |
| **Runtime** | holds every built component; the plugins' only shared state | `train/runtime.py`, built by `train/builder.py` |
| **Criterion** | a loss: declares what it needs (`features()`), returns numerators, never calls the model | `losses/base.py` |
| **FeatureRequest / ForwardRunner** | union of all needs → one forward; hooks capture only requested layers | `models/forward.py` |
| **LogitsView** | `[B,T,V]` logits kept lazy (chunked, gather, top-k) so memory stays low | `models/logits_view.py` |
| **Aggregator + Norms** | one global denominator per step (all micro-batches × ranks) so gradient accumulation is exact | `losses/aggregator.py` |
| **Teacher provider** | produces `TeacherOutput` (logits / top-k / hidden / embedding) | `teachers/` |
| **Stage / Artifact** | offline, content-hashed, resumable work (alignment, teacher caches, generation) | `pipeline/` |
| **Engine** | device, precision, DeepSpeed ZeRO/FSDP2, backward/step, checkpoint IO | `train/engine.py` |

Third-party libraries (DeepSpeed, vLLM, PEFT, transformers internals) are reached **only** through
[`src/lmkd/compat/`](../src/lmkd/compat/), guarded by `tests/unit/test_compat_canary.py`.

## 4. Suggested reading order (≈ 1 hour)

1. [README.md](../README.md) and one recipe: `recipes/same_tokenizer/fkl.yaml` (+ its `_base_`).
2. `losses/sft.py` (`nll`, the simplest criterion) → `losses/token_kd.py` (a real one).
3. `train/algorithms.py::Standard.loss`, then `train/step_strategies.py::StandardStep.run`.
4. `train/runtime.py` (`train_step`) and `train/builder.py::build_runtime` (the order everything is built).
5. `models/forward.py` and `models/logits_view.py` when you need to understand memory/features.
6. `examples/my_losses.py` and `examples/custom_loop.py` for the lightweight ways to extend.

## 5. I want to… (where to change)

| Goal | Do this |
|---|---|
| port a loss from a paper repo quickly | `@lmkd.register_loss` / `@lmkd.compute_loss` in one file ([examples/my_losses.py](../examples/my_losses.py)); load with `imports:` |
| add an optimized loss | `Criterion` subclass in `src/lmkd/losses/` (CONTRIBUTING.md §5.2), unit + parity test, recipe |
| new data processing | `Transform` (per sample) or `Stage` (offline, cached) |
| new way to produce batches (on-policy, replay) | `BatchProducer` (`data/producers.py`) |
| new optimizer | `OptimizerBuilder.build(groups)` in `train/optim.py`; assign it to a group in `optim.groups` |
| new optimizer *step* (SAM-like, multi-pass) | `StepStrategy`; reuse `accumulate` / `optimizer_step` |
| new metric | a pure function `metric(model, tokenizer, records) -> dict` (see `eval/embedding.py`), optionally wrapped in an `Evaluator` |
| use LM-KD loss math inside your own HF/TRL trainer | subclass the trainer, override `compute_loss`, call `lmkd.losses.functional` (pure tensor functions); no LM-KD wrapper exists |
| new model architecture | `ModelAdapter` in `models/adapters.py` (layers, lm_head, embeddings) |
| upgrade torch / transformers / DeepSpeed / vLLM / PEFT | change the pin, run `tests/unit/test_compat_canary.py`, fix `src/lmkd/compat/` |
| understand why a recipe fails | `lmkd check -c recipe.yaml` (static), `--preflight` (one real micro-batch) |

## 6. Conventions that keep it maintainable

- Plugin = one class + nested `Config` (strict) + docstring (first line is shown by `lmkd list`; add formula and
  `source=` of the original code).
- Types: `ruff check src tests examples` and `mypy` must pass (config in `pyproject.toml`; `pre-commit` runs both).
  The core (`core/`, `api`, `plugins`, `compat/`, `train/runtime`, `train/trainer`) is fully annotated.
- Keep functions short (limit enforced by `tests/unit/test_code_quality.py`); every public class has a docstring.
- Do not copy the training loop (every optimizer step goes through `rt.train_step`); do not touch private APIs of
  third-party libraries outside `compat/`.
- A change is done when `.venv/bin/python -m pytest -q` is green (CPU tests ≈ 20 s; GPU tests in `tests/gpu`).

## 7. Debugging tips

- A loss looks wrong: compare against the original code on a toy tensor (`tests/parity/`), then check the reduction
  (`token_sum` vs `mean`) and the target-aligned shift.
- Memory: look at `FeatureRequest` of your criteria (a `"full"` logits request materializes `[B,T,V]`).
- DeepSpeed hang: some engine methods are collective, every rank must call them (see `train/engine.py` docstring).
- Metrics: `rt.state.log("my/key", value)` from any plugin; they appear in `<output_dir>/metrics.jsonl`.
