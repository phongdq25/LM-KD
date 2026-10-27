# LM-KD Contributor Guide

> For a 15-minute high-level overview: read [ARCHITECTURE.md](ARCHITECTURE.md) first.
> Read this document for engineering standards, architectural contracts (C1–C17), and workflow guidelines.
> Consult `docs/DESIGN.md` for architectural rationale.

---

## 1. What You Are Building

LM-KD is a unified, plugin-based LLM distillation framework. **The core architecture is fully functional**:
configuration, registries, batch abstraction, collators, forward execution, losses, trainer, and CLI.
All plugins across the baseline suite are implemented, and the codebase contains 0 production stubs. A `stable` status
indicates architectural readiness rather than empirical replication of published benchmark numbers: see
[BASELINE_REPRODUCTION.md](BASELINE_REPRODUCTION.md). Contributions
must adhere to specifications and **must not break core invariants**.

Golden rules:
1. **Never duplicate the training loop.** The sole canonical loop body is `rt.train_step()` in
   `src/lmkd/train/runtime.py`; `Trainer.fit` simply invokes it. Standalone scripts in `examples/` or `scripts/`
   may implement custom loops, but each optimizer update must call `rt.train_step` (see §5.5). Official baselines
   must be expressed as recipe + plugin combinations. There are no exceptions: do not wrap external trainers
   (such as HF/TRL) inside LM-KD. To use a TRL trainer, subclass it externally and invoke `lmkd.losses.functional`
   within `compute_loss`.
2. **Do not modify core files** (enumerated in §7) unless accompanied by a thorough architectural justification.
3. **Test first, code second.** A contribution is complete only when the full test suite (`pytest -q`) passes cleanly.
4. **Adhere to original baseline implementations.** Every plugin faithfully reproduces a real baseline; inspect
   the reference source file specified in the docstring's metadata and paper citations.

---

## 2. Environment Setup & Daily Commands

```bash
cd lm-kd
python3 -m venv --system-site-packages .venv   # (reuse existing if present)
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q                  # run full test suite (CPU ~20s; tests/gpu for GPU tests)
.venv/bin/ruff check src tests examples        # linting: must be clean
.venv/bin/mypy                                 # type check: must be clean
.venv/bin/python -m pytest -q tests/unit/test_x.py::test_y   # run individual test
.venv/bin/lmkd list --status stub              # inspect planned stubs
.venv/bin/lmkd check -c recipes/<...>.yaml     # validate recipe
```

The environment **must** install the complete set of dependencies pinned in `pyproject.toml` (torch, transformers,
vllm, deepspeed, peft, lm_eval, liger, pytorch-optimizer, etc.). The framework does not support operating on a
reduced dependency subset.

External libraries are permitted and **encouraged** when they simplify code and enhance execution efficiency;
do not re-implement algorithms already provided by stable libraries. Re-implementation is justified only if an
external dependency contradicts architecture, incurs excessive weight, or requires only a few lines:
- Dependencies already listed in `[project.dependencies]`: import normally at the top of modules without `importorskip`.
- New dependencies: add to `[project] dependencies` with pinned versions and document in reports.
- Dependencies located strictly in `[project.optional-dependencies]` (e.g. `torchao`): use lazy imports inside
  functions and `pytest.importorskip` in tests. `pyproject.toml` is the source of truth: math-verify, rouge-score,
  scikit-learn, scipy, cvxpy, editdistance, and bitsandbytes are standard dependencies.
- Tests requiring a GPU must be marked with `@pytest.mark.gpu` and `skipif(not torch.cuda.is_available())`.

---

## 3. Codebase Map

```
src/lmkd/
  compat/              ALL external library version adaptations (ds_api, vllm_api, hf_api, trl_api) [CORE]
  integrations/        Reference backends for baseline reproduction: kto_runner, po_reference_runner
  api.py               lmkd.build / lmkd.run: library mode (object injection, custom loops)          [CORE]
  plugins.py           Automatic package discovery for lmkd.* + load_plugins(imports)                 [CORE]
  core/registry.py     Registry engine + all global registries (CRITERIONS, STAGES, ...)             [CORE]
  core/plugin.py       Plugin, BaseConfig (strict schema)                                            [CORE]
  core/config.py       YAML loading, _base_, --set, ${...}; RecipeConfig schema                      [CORE]
  core/batch.py        Batch container, FieldSpec, FIELDS + core fields                              [CORE]
  core/validate.py     lmkd check (requires/provides, stubs, teacher setup)                         [CORE]
  core/stubs.py        stub(...) helper
  data/labels.py       make_labels / supervised_mask (target-aligned convention)                     [CORE]
  data/collator.py     Generalized collator driven by FieldSpec                                      [CORE]
  data/sources.py      jsonl (+ hf_dataset, parquet)
  data/transforms/     Transform base + tokenize_sft (CANONICAL)
  data/producers.py    BatchProducer + dataset (CANONICAL) + mixed/adaptive_sgo/rollout_group
  data/catalog.py      Long-CoT segmenters/aligners/transforms
  models/adapters.py   HFAutoAdapter (layers, lm_head, backbone, adapters_disabled)
  models/logits_view.py LogitsView + chunked_token_divergence                                        [CORE]
  models/forward.py    SideRequest / FeatureRequest / SideOutput / ForwardRunner                     [CORE]
  models/modules.py    ModuleRegistry + linear/mlp projectors
  models/poolers.py    cls / mean / last poolers
  teachers/            signals (TeacherSignalSpec/Output), providers (hf_frozen),
                       input_builders (same/pretokenized), weight_policies
  losses/base.py       Criterion, LossTerm, LossOutput, LossContext                                  [CORE]
  losses/simple.py     Tier-0 (compute_loss) / Tier-1 (register_loss), ModelView                     [CORE]
  losses/aggregator.py Norms, weighted_sum, kd_mix                                                   [CORE]
  losses/sft.py        nll (CANONICAL REFERENCE), weighted_nll
  losses/token_kd.py   token_kd (fkl/rkl) ; functional/divergences.py
  losses/cross_tokenizer.py, losses/catalog.py   cross-tokenizer criteria
  train/engine.py      local (CANONICAL) + accelerate (DeepSpeed)
  train/optim.py       Parameter grouping rules, adamw/sgd/muon, schedulers
  train/grad.py        GradTransform clip_norm, gradient hooks
  train/algorithms.py  standard (CANONICAL) + rl algorithms
  train/step_strategies.py standard step (CANONICAL) + sam/asam/ppo_epochs
  train/callbacks.py   console_logger, nan_guard, checkpoint, evaluate
  train/builder.py     build_runtime: instantiation sequence + object injection                      [CORE]
  train/runtime.py     Runtime + begin/train_step/end/forward/evaluate/save                          [CORE]
  train/trainer.py     Trainer.fit = rt.begin; while: rt.train_step; rt.end                          [CORE]
  rollout/base.py      SamplingSpec, Rollout, rollouts_to_samples + engines
  pipeline/            artifacts (hashing), stage base, DAG runner, stage catalog
  eval/                metrics (pass@k), base evaluator/grader, embedding pure functions
  rl/catalog.py        verifiers and advantages
examples/              Single-file scripts: custom_loop.py, my_losses.py, recipes/
tests/
  conftest.py          Toy tokenizer + GPT-2 fixtures (student 2 layers d=32, teacher 3 layers d=48)
  unit/  e2e_tiny/  parity/
```

Registries look up a static AST index and import modules on demand; unindexed names fall back to full
package discovery (excluding `cli`, `scaffold`, and private `_` modules). New plugin files placed within
`src/lmkd/` require no manual registration edits. External user code registers via `imports:` in recipes
or CLI flags; `.py` file paths are resolved relative to the recipe YAML and imported idempotently.

### 3.1 Lifecycle of a Single Training Step

```
Trainer.fit
 └─ producer.next(state)                  -> Batch (for ONE optimizer step)
 └─ algorithm.augment(batch, rt)          -> teacher_input.build(...) (+ RL: reward/advantage)
 └─ step_strategy.run(batch, rt)
      ├─ micro_batches(batch)             -> split + .to(device)
      ├─ global_norms(micros)             -> Norms(z_tok, z_seq, n_micro) (all-reduce)
      ├─ accumulate(micros, norms)        -> per micro: algorithm.loss(micro) -> [WeightedOutput]
      │                                       aggregator.combine(...) -> loss ; backward
      ├─ apply_grad_transforms(metrics)   -> clip_norm, ...
      └─ optimizer_step()                 -> step all optimizers + schedulers, zero_grad
 └─ teacher.on_optimizer_step ; criterion.on_step_end ; callbacks.on_step_end
```

`Standard.loss(micro)` executes:
`ForwardRunner.run(rt.student_model, ..., req.student, grad=True, module=rt.student_module)`
→ `attach_grad_hooks(...)` (if `grad.hooks` configured) → `teacher.signals(micro, req.teacher_signal, req.teacher, student_view)`
→ `LossContext` → each criterion's `forward(ctx)`.

`req` = `Algorithm.request` = the union (`merge`) of `features()` declared by all criteria with currently non-zero
weights, merged with `features()` from active gradient hooks. The feature request is recomputed whenever the set of
active terms changes.

### 3.2 Common Framework APIs

| Requirement | Canonical Method |
|---|---|
| Forward student (always routes through engine wrapper) | `rt.student_module` (never call `rt.student_model(...)` during training) |
| Access underlying base model (save, lm_head, embedding) | `rt.student_model`, `rt.student_adapter` |
| Record auxiliary metric from any plugin | `rt.state.log("ns/key", value)` (or `ctx.state.log`) |
| Freeze student backbone, train auxiliary modules only | `models.student.trainable: false` |
| Teacher sees privileged prompt but identical response | `teacher.provider: {type: hf_frozen, position_map: response_shift}` |
| Attach hook to activation gradients | `grad.hooks: [{type: ..., target: "student.hidden[-8]"}]` |
| Stage consuming output from an upstream stage | `pipeline: - {stage: x, name: a} - {stage: y, inputs: {data: a}}` ; `Stage.attach(art, rt)` |
| Engine: unscale / optimizer step | `engine.unscale(opts)`, `engine.step_optimizers(opts) -> bool` (False = skipped fp16 step) |
| Producer checkpoint resume | `state_dict()/load_state_dict()`; use `EpochSampler` |
| Execute one full optimization step (from script/eval) | `rt.train_step(batch=None, loss_fn=None)` |
| HuggingFace-style forward outside criteria (eval, tier-0) | `rt.forward(batch, grad=False)` / `rt.teacher_forward(batch)` → `ModelView` |
| Evaluate all configured evaluators, update keep-best | `rt.evaluate()` → `rt.state.extra["eval"]` |

---

## 4. Mandatory Architectural Contracts (Cheat Sheet)

| # | Rule | Correct Example |
|---|---|---|
| C1 | Every plugin must define `class Config(BaseConfig)`; constructor accepts **only** `cfg`. Runtime dependencies must be acquired in `setup(rt)` or context. | `def setup(self, rt): self.tok = rt.student_tokenizer` |
| C2 | Strict Config: no `**kwargs`; fields named `type` or `params` are forbidden. Default values must faithfully match original baseline code. | `temperature: float = 2.0` |
| C3 | Batch keys must be namespaced `ns.name`. Plugins in `src/` register `FIELDS.register(FieldSpec(...))` **at module top level**. Unregistered keys are auto-inferred with a warning and 0-padded; explicit registration is **mandatory** when pad value ≠ 0. | `FIELDS.register(FieldSpec("align.eta_s_idx", "ragged", torch.long))` |
| C4 | Explicitly declare `requires` and `provides` (or override `required_keys(cfg)` / `provided_keys(cfg)` when config-dependent). | `requires = ("seq.token_weights",)` |
| C5 | **Target-aligned convention**: per-token tensors have sequence length T; index t corresponds to `input_ids[t]`. Labels are created exclusively via `make_labels`. Logits are queried via `LogitsView` (`target_logp`, `chunks`, `gather`, `topk`, `entropy`, `at`). `logits[:, :-1]` is **strictly forbidden** in criteria. | `ctx.student.logits.target_logp(labels)` |
| C6 | Criterion returns `LossTerm(numerator, reduction)`. Sum over tokens → `"token_sum"`; sum over sequences → `"seq_sum"`; already normalized within micro-batch → `"mean"`. **Never** manually divide by batch size or token count. | `LossTerm((per_tok * mask).sum(), "token_sum")` |
| C7 | Criteria **never execute forward passes directly**. Declare needed representations via `features()`. (Tier-0/1 losses — `compute_loss`/`register_loss` — are reserved for rapid porting, examples, and parity tests; in-tree plugins in `src/` are Tier-2). | `SideRequest(logits="view", hidden=(-1,))` |
| C8 | Trainable auxiliary parameters must be created exclusively via `modules.add(name, module, group=..., owner=self.alias)` within `build_modules`. Naming convention: `f"{alias}__{name}"` (`.` auto-replaces with `__`). Retrieve via `ctx.modules.get(name)`. | See §5.2 |
| C9 | `[B,T,V]` fp32 tensors may only be accessed via `LogitsView.full()`, accompanied by a code comment stating justification. Prefer `chunked_token_divergence`. | |
| C10 | Expensive, deterministic operations (token alignment, scoring, attribution) belong in **offline Stages/Transforms**, never in `Criterion.forward`. | |
| C11 | Do not mutate tensors inside `Batch` or `SideOutput` in-place. If modifications are required, invoke `.clone()`. | |
| C12 | Do not hardcode token IDs or architecture names. Retrieve values dynamically via the tokenizer and `rt.student_adapter`. | |
| C13 | Class docstring: line 1 = concise summary; subsequent lines describe mathematical formulation + original source path. Specify `source=` / `paper=` upon registration. | |
| C14 | New plugin modules placed in `src/lmkd/` are discovered automatically without registration side effects. External code uses `imports:` in YAML recipes or `--plugins`. | |
| C15 | Evaluation metrics are **pure functions** `(model, tokenizer, records, **opts) -> dict`; `Evaluator.evaluate(rt)` loads records from Config, calls the pure function on `rt.student_model`, and adds key prefixes. Reference: `eval/embedding.py`. | `sts_spearman(model, tok, records)` |
| C16 | Never access private/internal APIs of external libraries (torch, transformers, deepspeed, vllm, peft) outside `src/lmkd/compat/`. If a new compatibility assumption is required, expose it through `compat/` **and** add a canary test in `tests/unit/test_compat_canary.py`. `tests/unit/test_code_quality.py` enforces this. | `ds_api.is_partitioned_param(p)` |
| C17 | Code quality and typing: `ruff` and `mypy` must be completely clean. Public classes must have docstrings (the first line is displayed by `lmkd list`); functions should remain modular (≤ 65 lines); `self.cfg` is typed as `Any` due to nested `class Config` patterns. | |

---

## 5. Plugin Templates by Category

### 5.1 Transform (Offline / In-Memory)

```python
# src/lmkd/data/transforms/my_transform.py
import torch
from lmkd.core.batch import FIELDS, FieldSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TRANSFORMS
from lmkd.data.transforms.base import Transform

FIELDS.register(FieldSpec("seq.my_weight", "token", torch.float32, 1.0, "student"))

@TRANSFORMS.register("my_transform", source="baseline/path.py:10-40")
class MyTransform(Transform):
    """One-line summary.  Formula: ...  Source: ..."""
    requires = ("student.input_ids", "student.labels")
    provides = ("seq.my_weight",)

    class Config(BaseConfig):
        alpha: float = 1.0

    def __call__(self, sample, ctx):
        n = len(sample["student.input_ids"])
        sample["seq.my_weight"] = [self.cfg.alpha] * n      # list, np array, or tensor
        return sample                                       # return None to drop sample
```

### 5.2 Criterion with Learnable Module

For lightweight losses without auxiliary modules, `@lmkd.loss` operating on target-aligned `Inputs` is recommended.
Masking contracts, chunking modes, and offline test examples are documented in [NEW_METHOD_GUIDE.md](NEW_METHOD_GUIDE.md).
Baselines involving projectors, cross-token alignments, or persistent state inherit from `Criterion` and use the lifecycle below:

```python
@CRITERIONS.register("my_feature_kd", source="...")
class MyFeatureKD(Criterion):
    """mean_t (1 - cos(P h_S, h_T)) on designated hidden layers."""
    needs_teacher = True

    class Config(BaseConfig):
        layer_s: int = -1
        layer_t: int = -1

    def features(self):
        return FeatureRequest(
            student=SideRequest(hidden=(self.cfg.layer_s,)),
            teacher=SideRequest(hidden=(self.cfg.layer_t,)),
            teacher_signal=TeacherSignalSpec(kind="hidden"),
        )

    def build_modules(self, info, modules):
        cfg = LinearProjector.Config(in_dim=info.student_hidden, out_dim=info.teacher_hidden)
        modules.add(f"{self.alias}__proj", LinearProjector(cfg), group="projector", owner=self.alias)

    def forward(self, ctx):
        hs = ctx.student.hidden(self.cfg.layer_s)
        ht = ctx.teacher.side.hidden(self.cfg.layer_t)
        p = ctx.modules.get(f"{self.alias}__proj")(hs)
        m = ctx.batch["student.attention_mask"].float()
        cos = F.cosine_similarity(p.float(), ht.float(), dim=-1)
        return LossOutput({"cos": LossTerm(((1 - cos) * m).sum(), "token_sum")})
```

### 5.3 Offline Stage (DAG Node)

```python
@STAGES.register("my_stage", source="...")
class MyStage(Stage):
    class Config(BaseConfig):
        model: str

    code_version = "1"                       # bump whenever output semantics change

    def run(self, ctx):
        prev = ctx.inputs.get("prev")        # Upstream stage Artifact (or None)
        out = ctx.output.path
        out.mkdir(parents=True, exist_ok=True)
        # process records; support per-sample resume if runtime is long
        return {"n_rows": n}                 # saved into artifact manifest

    def load_samples(self, art):             # required only when stage serves as dataset source
        return [json.loads(line) for line in open(art.path / "samples.jsonl")]
```

### 5.4 BatchProducer, StepStrategy, GradTransform, Callback, Optimizer

- **Producer**: Subclass `BatchProducer`; in `setup(rt)` instantiate DataLoader and state buffers; `next(state)` returns a `Batch` for **one** optimizer step (`train.batch_size / world_size` rows). Rollouts route via `rollouts_to_samples` → `rt.collator`. Implement `state_dict`/`load_state_dict` for checkpoint resumption.
- **StepStrategy**: Subclass `StepStrategy`; reuse canonical building blocks: `micro_batches`, `global_norms`, `accumulate`, `apply_grad_transforms`, `optimizer_step`. For example, SAM executes: `accumulate` (phase 1) → perturb weights → `accumulate` (phase 2) → restore weights → `optimizer_step`.
- **GradTransform**: `__call__(params: list[ParamWithGroup], ctx: StepContext)` mutates `p.param.grad` in-place and logs diagnostics into `ctx.metrics`.
- **Callback**: Override required lifecycle hooks; lower `order` runs earlier.
- **Optimizer**: `@OPTIMIZERS.register("x") class XBuilder(OptimizerBuilder): def build(self, groups): ...` (strip `"names"` via `_strip(groups)`).

### 5.5 Tier-0/1 Losses & Library Mode (DESIGN §6.7, §13.4)

Ideal for rapid porting or reference verification in parity tests; runnable example in `examples/my_losses.py`:

```python
import lmkd

@lmkd.register_loss("my_fkl", teacher=True, reduction="token_sum")   # Tier 1 = standard criterion
def my_fkl(s, t, batch, temperature: float = 1.0):                    # s/t: ModelView, HF-style logits
    mask = (batch["student.labels"][:, 1:] != -100).float()
    per_tok = -(t.logits[:, :-1].div(temperature).softmax(-1)
                * s.logits[:, :-1].div(temperature).log_softmax(-1)).sum(-1)
    return (per_tok * mask).sum()

@lmkd.compute_loss("my_ce")                                            # Tier 0: manages own forward
def my_ce(batch, rt):
    s = rt.forward(batch)
    labels = batch["student.labels"][:, 1:]
    return torch.nn.functional.cross_entropy(s.logits[:, :-1].flatten(0, 1), labels.flatten(), ignore_index=-100)
```

Library mode usage: `rt = lmkd.build(recipe, ...)`, followed by `rt.begin(); while not rt.done: rt.train_step(loss_fn=...); rt.end()`.
Never invoke `backward` or `optimizer.step` outside `rt.train_step`. To use TRL trainers, subclass them and call
pure-tensor primitives from `lmkd.losses.functional`.

---

## 6. Testing Requirements

### 6.1 Required Test Tiers per Plugin

1. **Unit tests** (`tests/unit/test_<module>.py`): Fast CPU tests (< 2s) on small tensors verifying mathematical formulas against manual calculations or simple loop baselines.
2. **Parity tests** (`tests/parity/test_<method>_parity.py`, `@pytest.mark.parity`): Numerical comparison against original baseline implementations on identical random tensors, with tolerance `atol=1e-5` (fp32).
   - Pattern 1: Add baseline repository to `sys.path` and import functions directly (using `pytest.importorskip` for optional dependencies).
   - Pattern 2: When external dependencies prevent clean import, copy the original function into `tests/parity/_ref/<method>_ref.py` with citation comments and line numbers.
   - Known baseline bugs (DESIGN Appendix A): Fix bugs in the reference copy with comment `# FIX(original bug): ...`.
3. **End-to-End tests** (`tests/e2e_tiny/`): Use `base_recipe` and `tiny_models` fixtures; execute 3–5 training steps verifying finite losses and metric emission.

### 6.2 Predefined Test Fixtures

- `tiny_models`: Provides `{"student": path, "teacher": path}` containing tiny 2-layer / 3-layer GPT-2 models sharing a character/word vocabulary (vocab size 67).
- `toy_jsonl`: 32 rows containing `{id, prompt, response}`.
- `base_recipe`: Standard SFT recipe dictionary; mutate and pass to `recipe_from_dict`.
- Custom tokenizers (for cross-tokenizer testing): instantiate using `_make_tokenizer` helper in `conftest.py`.

---

## 7. Core Files — DO NOT Modify Without Authorization

`api.py`, `plugins.py`, `compat/*`, `core/registry.py`, `core/plugin.py`, `core/config.py`, `core/batch.py`, `core/validate.py`,
`data/labels.py`, `data/collator.py`, `models/logits_view.py`, `models/forward.py`, `losses/base.py`, `losses/simple.py`,
`losses/aggregator.py`, `train/builder.py`, `train/runtime.py`, `train/trainer.py`, `train/step_strategies.py::StandardStep`,
`train/engine.py::Engine`, `train/algorithms.py::Algorithm`, `pipeline/runner.py`, `models/adapters.py::HiddenForward`,
`tests/e2e_tiny/test_skeleton_fixes.py` (regression test suite — must never be weakened).

If a modification to core files is genuinely necessary: **stop**, write a `## CORE CHANGE REQUEST` section in your
report detailing the rationale, proposed diff, and impact analysis.

Permitted extensions: register new fields via `FIELDS.register`, register new plugins, add module-private helpers,
or create standalone scripts in `examples/`.

---

## 8. Common Pitfalls & Prevention

| Pitfall | Consequence | Prevention |
|---|---|---|
| Manually shifting `logits[:, :-1]` alongside `labels[:, 1:]` | 1-position alignment offset against LogitsView | Always use `LogitsView` (C5) |
| Filtering lengths using `ids != pad_id` | Broken when `pad == eos` (e.g. Qwen) | Use `attention_mask` or recorded lengths |
| Dividing loss by `mask.sum()` in criterion | Skews gradients under gradient accumulation | Return `LossTerm(..., "token_sum")` (C6) |
| Using `F.kl_div(reduction="batchmean")` on `[B,T,V]` | Includes padding in normalization, incorrect scale | Compute per-token and multiply mask |
| Instantiating `nn.Linear` in criterion `__init__` | Parameters not optimized or saved | Register via `modules.add` in `build_modules` (C8) |
| Storing step/epoch state in global variables | Checkpoints fail to resume deterministically | Use `ctx.state.step`, `ctx.state.progress` |
| Forgetting `FIELDS.register` for a new batch field | Collator falls back to 0-padding with warning; fails for −100 pad | Register at module top level (C3) |
| `'x' is not registered` error on external plugins | File has not been imported | Provide `imports:` in recipe or `--plugins` |
| Calling `rt.student_model(...)` in eval or tier-0 loss | Breaks under DeepSpeed ZeRO-3 | Use `rt.forward(batch, grad=False)` |
| Leaving duplicate `stub(...)` with same name | Raises `already registered` error | Remove stub upon implementing real class |
| Hardcoded constants omitted from Config | Prevents hyperparameter tuning / replication | Expose in Config with original default values |

---

## 9. Development & Contribution Workflow

1. Discuss your proposed feature or fix in a GitHub issue.
2. Run `pytest -q` to confirm the existing test suite is fully passing.
3. Write failing unit/parity tests corresponding to the required specification.
4. Implement the class under the registered name, removing any corresponding `stub(...)`.
5. Execute the targeted test file, then run the full suite (`pytest -q`).
6. Add or update YAML recipes in `recipes/<domain>/` if specified; validate with `lmkd check`.
7. Ensure linting and type checking pass (`ruff check src tests examples` and `mypy`).
8. Submit a Pull Request following the template below.

---

## 10. Pull Request Description Template

```markdown
### Summary
- Feature / plugin implemented: [criterions] dskdv2, ...
- Files: src/lmkd/..., tests/...
- Tests: N passed (unit X, parity Y, e2e Z); command: ...
- Parity: max abs diff = ...
- Deviations from original code (and rationale): ...
- Known limitations: ...
- Architectural changes (if any): none | <details>
```

## 11. Definition of Done Checklist

- [ ] All stubs replaced with concrete classes; `lmkd list` reports `stable`.
- [ ] Config defines all parameters with defaults matching original code; docstring includes formula and source references.
- [ ] Invariants `requires` / `provides` / `features()` accurately declared; new batch keys registered via `FIELDS.register`.
- [ ] Unit tests and parity tests pass; full test suite (`pytest -q`) passes.
- [ ] `ruff check src tests examples` and `mypy` are clean with zero errors.
- [ ] Core framework invariants preserved.
- [ ] Recipe passes `lmkd check`.
- [ ] Code is formatted and documented cleanly.
