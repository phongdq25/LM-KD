# Developing and Testing a New Method

Get started with an offline CPU smoke test:

```bash
.venv/bin/python examples/new_method.py --steps 2 --output /tmp/lmkd-new-method
.venv/bin/lmkd try example_confident_kd --plugins examples/new_method.py
```

The [runnable example](../examples/new_method.py) initializes a custom tokenizer and two small GPT-2 models, registers a KD
loss equipped with a confidence mask, combines it with NLL, and invokes `lmkd.build(...).fit()` to train and save
checkpoints. It verifies integration mechanics; randomly initialized weights do not benchmark empirical performance.

## Choosing Extension Points

| Capability Required | Recommended API |
|---|---|
| Logit-level loss, identical tokenizers, no learned auxiliary parameters | `@lmkd.loss`, returns `TokenLoss` |
| Sequence-level loss, cross-token interactions, or whole-batch operations | `@lmkd.loss(..., chunked=False)`, returns `SeqLoss`/`ScalarLoss`/dict |
| Hidden-state access only | `@lmkd.loss(..., logits=False, capture={"s": [1]})` |
| Projectors, alignments, multiple student views, cached teachers, or resumable states | `Criterion`: declare `features`, `requires`, create submodules in `build_modules`, initialize via `prepare` |
| New data fields | `Transform` and `FieldSpec`; verify via preflight on a toy batch |
| Custom optimizer step mechanics, such as SAM | `StepStrategy`, using engine contracts; SAM/PCGrad currently support local engine |

Legacy APIs `register_loss` and `compute_loss` remain supported for backward compatibility. `register_loss` feeds
HuggingFace-convention logits into the function; `@lmkd.loss` provides target-aligned logits. Do not shift logits
an extra time when porting from legacy code.

## Essential Loss Contracts

- `s.logits[:, t]` predicts `s.labels[:, t]`. Target position 0 is not supervised; labels `-100` are ignored. Hidden
  states adhere to HuggingFace indexing conventions; `capture` uses HF hidden layer indices: 0 is the embedding layer,
  1..L are transformer blocks, and negative integers count from the final layer.
- Within a chunk, `s.start:s.end` represents the slice relative to the full sequence. `s.labels` and `s.mask` are
  already sliced; `s.batch` remains full. Custom added fields must be sliced using those offsets.
- `TokenLoss(values, mask, log_values)` accepts shapes `[B,T]` or `[B,t]`. Custom masks intersect with the supervision
  mask; omitted tokens cannot be re-enabled. The denominator remains the total supervised token count across the entire
  step, not merely the tokens selected by a confidence mask.
- `SeqLoss` takes `[B]` and normalizes by sequence count; `ScalarLoss` takes a scalar averaged across micro-batches and
  ranks. Do not use scalar means when exact token-averaged metrics across heterogeneous micro-batches are required.
- By default, `chunked=None` automatically falls back to full-sequence computation if the function returns sequence/scalar/dict
  or accesses student hidden/side representations. The function may be evaluated once on a chunk before switching. Use
  `chunked=False` when the loss has side effects, token-token interactions, RNG state, or higher-order gradients. A
  `[B,t]` tensor does not automatically guarantee token-locality.
- The recomputation path is reserved for deterministic, token-local operations on logits, labels/masks, and non-gradient
  tensors. It must not close over trainable parameters; use `Criterion` for modules with projectors. `chunked=True`
  strictly requires this contract.
- `logits` are evaluated lazily on the full-sequence path. Setting `logits=False` bypasses LM head computation entirely.
  Sequence packing is opt-in (`supports_packing=True`), enabled only when the loss respects sample boundaries and
  avoids mixing packed sequences.

## From Prototyping to Real-World Training

1. Write unit tests for loss computation and gradient flow on small synthetic tensors: heterogeneous masks, `-100` padding,
   multiple chunks, and varied shapes; verify full vs. chunked equivalence if targeting recomputation.
2. `lmkd try <name> --plugins my_loss.py` verifies finite loss and student parameter gradient updates. Synthetic probes
   do not simulate full lifecycles, alignments, or projectors; losses requiring these must be tested with `Runtime` as
   demonstrated in `examples/new_method.py`.
3. Construct a YAML recipe referencing `imports: [my_loss.py]` (path relative to the YAML file), models, datasets,
   transforms, loss terms, and training budget. Run `lmkd check -c ...` for static schema and dependency validation;
   use `--preflight` to run a dry-run step on actual data and model weights.
4. Execute 1–2 steps on a local engine, verifying that auxiliary projectors update and checkpoints resume correctly.
   Scale up batch size, enable mixed precision and packing progressively, and finally transition to distributed engines.

`lmkd new criterion my_method` scaffolds a baseline NLL setup, a starter recipe, and a working gradient test. Recipes
are self-contained without mandatory base inheritance; specify your model paths and populate `data/train.jsonl` with
`prompt`/`response` records before training.

## Deployment Limits

FSDP2 operates under the `fsdp2` engine registry, requiring CUDA and `torchrun`. Optimizers are constructed after
sharding. Validation covers dual GPUs with two AdamW optimizers, auxiliary modules, gradient accumulation, global norm
clipping, and full checkpoint save/resume. Use `reshard_after_forward: false` because auxiliary/logit evaluation takes
place after backbone forward passes; the full model gathers across forward/backward passes.
SGD without clipping has also been verified against local execution for gradient scaling parity. Other optimizers must
be verified for DTensor compatibility; SAM/PCGrad and adapter-only PEFT checkpoints are currently unsupported (PEFT
is rejected early). Rollouts, self-teachers, and methods requiring forward passes outside `Runtime` have not yet been
validated on FSDP2.

LM-KD does not wrap external training loops. To integrate custom losses into HF or TRL trainers, subclass the external
trainer, override `compute_loss`, and invoke pure-tensor primitives from `lmkd.losses.functional`. For single-forward
RL/on-policy distillation with vLLM and ZeRO/FSDP2, use LM-KD's native training loop via YAML recipes.
