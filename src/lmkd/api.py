"""Library mode: build a run from a recipe (+ Python objects) and drive it from your own script.

    import lmkd

    rt = lmkd.run("recipes/same_tokenizer/fkl.yaml", overrides=["train.max_steps=100"])   # = lmkd train

    # ... or your own loop, reusing every step semantic of the framework
    rt = lmkd.build(recipe_dict, student_model=my_model, student_tokenizer=tok, train_records=rows)
    rt.begin()
    while not rt.done:
        out = rt.train_step(loss_fn=my_loss)   # my_loss(micro_batch, rt) -> Tensor | dict | LossOutput
        if rt.state.step % 100 == 0:
            print(rt.evaluate())               # recipe evaluators + callables fn(rt) -> dict
    rt.end()
    rt.save("final")

Inside a loss function: ``rt.forward(batch)`` (student, through the engine wrapper: DeepSpeed
safe) and ``rt.teacher_forward(batch)`` return :class:`lmkd.losses.simple.ModelView`.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lmkd.core.config import RecipeConfig, load_yaml_with_bases, recipe_from_dict
from lmkd.train.runtime import Runtime

INJECTED_PATH = "<injected>"


def _raw_recipe(recipe: str | Path | dict | RecipeConfig) -> dict:
    if isinstance(recipe, RecipeConfig):
        return recipe.model_dump()
    if isinstance(recipe, dict):
        return copy.deepcopy(recipe)
    return load_yaml_with_bases(recipe)


def build(
    recipe: str | Path | dict | RecipeConfig,
    *,
    overrides: list[str] | None = None,
    imports: list[str] | None = None,
    student_model: Any = None,
    student_tokenizer: Any = None,
    teacher_model: Any = None,
    teacher_tokenizer: Any = None,
    train_records: list[dict] | None = None,
    loss_fn: Callable | None = None,
    evaluators: list[Any] | None = None,
    check: bool = True,
) -> Runtime:
    """Build a :class:`Runtime` (models, data, optimizers, engine, callbacks) ready to train.

    Args:
        recipe: YAML path, dict or RecipeConfig. Sections given as objects below may be omitted.
        overrides: dotted overrides like the CLI ``--set`` (``["train.max_steps=10"]``).
        imports: extra plugin files/modules (added to the recipe's ``imports:``).
        student_model / student_tokenizer / teacher_model / teacher_tokenizer: use these objects
            instead of loading ``models.*.path``.
        train_records: raw records (dicts) passed through ``data.transforms`` instead of ``data.train``.
        loss_fn: tier-0 loss ``fn(micro_batch, rt)`` used at every step (added to ``loss.terms``).
        evaluators: extra evaluators, Evaluator plugins or callables ``fn(rt) -> dict``.
        check: run the static recipe check first (raises ValueError with the report).
    """
    from lmkd.core.validate import check_recipe
    from lmkd.train.builder import build_runtime

    raw = _raw_recipe(recipe)
    if imports:
        raw["imports"] = list(dict.fromkeys([*(raw.get("imports") or []), *imports]))
    if student_model is not None:
        if student_tokenizer is None:
            raise ValueError("student_model= needs student_tokenizer=")
        raw.setdefault("models", {}).setdefault("student", {}).setdefault("path", INJECTED_PATH)
    cfg = recipe_from_dict(raw, overrides)

    injected = frozenset(k for k, v in {"train_records": train_records, "loss_fn": loss_fn}.items() if v is not None)
    if check:
        rep = check_recipe(cfg, injected=injected)
        if not rep.ok:
            raise ValueError("recipe check failed:\n" + rep.render())
    rt = build_runtime(cfg, student_model=student_model, student_tokenizer=student_tokenizer,
                       teacher_model=teacher_model, teacher_tokenizer=teacher_tokenizer,
                       train_records=train_records, loss_fn=loss_fn)
    rt.evaluators.extend(evaluators or [])
    return rt


def run(recipe: str | Path | dict | RecipeConfig, **kwargs: Any) -> Runtime:
    """``build(...)`` then the standard loop (``rt.fit()``) = ``lmkd train``."""
    return build(recipe, **kwargs).fit()


def _parse_quick_loss_terms(loss: str | Callable | list[Any]) -> list[dict[str, Any]]:
    """Parse string, callable, dict, tuple or list into standardized recipe loss terms."""
    items: list[Any] = [loss] if not isinstance(loss, list) else list(loss)
    terms = []
    for item in items:
        if isinstance(item, str):
            terms.append({"criterion": item})
        elif callable(item):
            name = getattr(item, "registry_name", None)
            if name is None:
                raise TypeError("quick loss callables must be registered with @lmkd.loss or @register_loss")
            terms.append({"criterion": name})
        elif isinstance(item, dict):
            terms.append(item)
        elif isinstance(item, tuple) and len(item) == 2:
            terms.append({"criterion": str(item[0]), "weight": float(item[1])})  # type: ignore[dict-item]
        else:
            raise TypeError(f"unsupported quick loss item: {type(item).__name__}")
    if not terms:
        raise ValueError("quick loss must contain at least one term")
    return terms


def _build_quick_recipe(
    student: str | Any,
    data: str | list[dict] | Path,
    loss: str | Callable | list[Any],
    teacher: str | Any | None,
    output_dir: str,
    max_steps: int,
    lr: float,
    batch_size: int,
    micro_batch_size: int,
    engine: str,
) -> tuple[dict[str, Any], list[dict] | None]:
    """Construct an in-memory recipe dict and optional raw dataset records."""
    terms = _parse_quick_loss_terms(loss)
    recipe: dict[str, Any] = {
        "engine": {"type": engine},
        "models": {
            "student": {"path": student if isinstance(student, (str, Path)) else INJECTED_PATH},
        },
        "data": {"transforms": [{"type": "tokenize_sft"}]},
        "train": {
            "output_dir": output_dir,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
        },
        "optim": {"lr": lr, "optimizers": {"default": {"type": "adamw"}}},
        "loss": {"terms": terms},
    }
    if teacher is not None:
        recipe["models"]["teacher"] = {"path": teacher if isinstance(teacher, (str, Path)) else INJECTED_PATH}
        recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}

    records = None
    if isinstance(data, (str, Path)):
        recipe["data"]["train"] = {"type": "jsonl", "path": str(data)}
    elif isinstance(data, list):
        records = data
        recipe["data"]["train"] = {"type": "jsonl", "path": "in_memory.jsonl"}
    return recipe, records


def quick(
    student: str | Any,
    data: str | list[dict] | Path,
    loss: str | Callable | list[Any] = "nll",
    *,
    teacher: str | Any | None = None,
    output_dir: str = "outputs/quick",
    max_steps: int = 100,
    lr: float = 5e-5,
    batch_size: int = 2,
    micro_batch_size: int = 1,
    engine: str = "local",
    strict: bool = False,
    **kwargs: Any,
) -> Runtime:
    """Rapid prototyping API (v2): build a complete training Runtime directly from Python
    without writing a recipe YAML file.

    Args:
        student: Model path or HuggingFace ID (or nn.Module object).
        data: Path to JSONL file, or list of raw record dicts.
        loss: Loss name (e.g. "nll", "fkl"), loss function decorated with `@lmkd.loss`,
            or list of terms.
        teacher: Optional teacher model path or ID (or nn.Module).
        output_dir: Directory where checkpoints and metrics are saved.
        max_steps: Total training steps.
        lr: Learning rate.
        batch_size: Total batch size per optimizer step.
        micro_batch_size: Micro-batch size per forward-backward.
        engine: "local", "accelerate", or "fsdp2".
        strict: If False, skips rigid static check allowing quick iteration.
    """
    recipe_dict, records = _build_quick_recipe(
        student, data, loss, teacher, output_dir, max_steps, lr, batch_size, micro_batch_size, engine
    )
    student_model = student if not isinstance(student, (str, Path)) else None
    teacher_model = teacher if not isinstance(teacher, (str, Path)) else None

    return build(
        recipe_dict,
        student_model=student_model,
        teacher_model=teacher_model,
        train_records=records,
        check=strict,
        **kwargs,
    )
