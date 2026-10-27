"""Scaffold generator for new plugins."""
from __future__ import annotations

import sys
from pathlib import Path

_KINDS = {
    "criterion": {
        "module": "losses",
        "reg": "CRITERIONS",
        "import_stmt": "from lmkd.core.registry import CRITERIONS\nfrom lmkd.losses.sft import NLL",
        "base_cls": "Criterion",
        "code": """@CRITERIONS.register("{name}", status="experimental")
class {cls_name}(NLL):
    '''Runnable token-NLL baseline; replace forward to implement your method.'''

    def forward(self, ctx):
        return super().forward(ctx)

""",
    },
    "transform": {
        "module": "data/transforms",
        "reg": "TRANSFORMS",
        "import_stmt": "from lmkd.core.registry import TRANSFORMS\nfrom lmkd.core.plugin import BaseConfig\nfrom lmkd.data.transforms.base import Transform",
        "base_cls": "Transform",
        "code": """@TRANSFORMS.register("{name}", status="experimental")
class {cls_name}(Transform):
    '''Extension starting point; document the method contract here.'''

    class Config(BaseConfig):
        pass

    def __call__(self, sample, ctx):
        return sample
""",
    },
    "stage": {
        "module": "pipeline/stages",
        "reg": "STAGES",
        "import_stmt": "from lmkd.core.registry import STAGES\nfrom lmkd.core.plugin import BaseConfig\nfrom lmkd.pipeline.stage import Stage",
        "base_cls": "Stage",
        "code": """@STAGES.register("{name}", status="experimental")
class {cls_name}(Stage):
    '''Extension starting point; document the method contract here.'''

    class Config(BaseConfig):
        pass

    def run(self, ctx):
        raise NotImplementedError
""",
    },
    "producer": {
        "module": "data/producers_ext",
        "reg": "PRODUCERS",
        "import_stmt": "from lmkd.core.registry import PRODUCERS\nfrom lmkd.core.plugin import BaseConfig\nfrom lmkd.data.producers import BatchProducer",
        "base_cls": "BatchProducer",
        "code": """@PRODUCERS.register("{name}", status="experimental")
class {cls_name}(BatchProducer):
    '''Extension starting point; document the method contract here.'''

    class Config(BaseConfig):
        pass

    def next(self, state):
        raise NotImplementedError
""",
    },
    "step_strategy": {
        "module": "train/steps_ext",
        "reg": "STEP_STRATEGIES",
        "import_stmt": "from lmkd.core.registry import STEP_STRATEGIES\nfrom lmkd.core.plugin import BaseConfig\nfrom lmkd.train.step_strategies import StepStrategy",
        "base_cls": "StepStrategy",
        "code": """@STEP_STRATEGIES.register("{name}", status="experimental")
class {cls_name}(StepStrategy):
    '''Extension starting point; document the method contract here.'''

    class Config(BaseConfig):
        pass

    def run(self, batch, rt):
        raise NotImplementedError
""",
    },
    "grad_transform": {
        "module": "train/grad_ext",
        "reg": "GRAD_TRANSFORMS",
        "import_stmt": "from lmkd.core.registry import GRAD_TRANSFORMS\nfrom lmkd.core.plugin import BaseConfig\nfrom lmkd.train.grad import GradTransform",
        "base_cls": "GradTransform",
        "code": """@GRAD_TRANSFORMS.register("{name}", status="experimental")
class {cls_name}(GradTransform):
    '''Extension starting point; document the method contract here.'''

    class Config(BaseConfig):
        pass

    def __call__(self, params, ctx):
        pass
""",
    },
    "callback": {
        "module": "train/callbacks_ext",
        "reg": "CALLBACKS",
        "import_stmt": "from lmkd.core.registry import CALLBACKS\nfrom lmkd.core.plugin import BaseConfig\nfrom lmkd.train.callbacks import Callback",
        "base_cls": "Callback",
        "code": """@CALLBACKS.register("{name}", status="experimental")
class {cls_name}(Callback):
    '''Extension starting point; document the method contract here.'''

    class Config(BaseConfig):
        pass
""",
    },
}


def create_scaffold(kind: str, name: str, domain: str = "experimental", root: Path | None = None) -> int:
    if kind not in _KINDS:
        sys.stderr.write(f"Unknown kind '{kind}'. Must be one of: {sorted(_KINDS)}\n")
        return 1

    root = root or Path.cwd()
    info = _KINDS[kind]
    cls_name = "".join(part.capitalize() for part in name.split("_"))

    plugin_file = root / "src" / "lmkd" / info["module"] / f"{name}.py"
    test_file = root / "tests" / "unit" / f"test_{name}.py"
    recipe_file = (root / "recipes" / domain / f"{name}.yaml") if kind == "criterion" else None

    targets = [plugin_file, test_file]
    if recipe_file is not None:
        targets.append(recipe_file)

    for p in targets:
        if p.exists():
            sys.stderr.write(f"File already exists: {p}\n")
            return 1

    # Create directories
    for p in targets:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Create __init__.py if inside src/lmkd and not existing
        if "src/lmkd" in str(p.parent) and not (p.parent / "__init__.py").exists():
            (p.parent / "__init__.py").touch()

    # Write plugin file
    plugin_content = f"{info['import_stmt']}\n\n\n{info['code'].format(name=name, cls_name=cls_name)}"
    plugin_file.write_text(plugin_content, encoding="utf-8")

    # Write test file
    test_content = (
        "from pathlib import Path\n"
        "\n"
        f"from lmkd.core.registry import {info['reg']}\n"
        "from lmkd.plugins import load_plugins\n\n"
        f"load_plugins([str(Path(__file__).resolve().parents[2] / 'src/lmkd/{info['module']}/{name}.py')])\n\n\n"
        f"def test_{name}_build():\n"
        f"    assert {info['reg']}.build('{name}') is not None\n"
    )
    if kind == "criterion":
        test_content += (
            f"\n\ndef test_{name}_gradient():\n"
            "    from lmkd.scaffold.probe import probe_gradient\n"
            f"    probe_gradient(CRITERIONS.build('{name}'))\n"
        )
    test_file.write_text(test_content, encoding="utf-8")

    # Write recipe file if criterion
    if recipe_file is not None:
        recipe_content = (
            f"imports: [../../src/lmkd/{info['module']}/{name}.py]\n"
            "engine: {type: local, device: cpu, mixed_precision: 'no'}\n"
            "models:\n  student: {path: sshleifer/tiny-gpt2}\n"
            "data:\n  train: {type: jsonl, path: data/train.jsonl}\n"
            "  transforms: [{type: tokenize_sft, use_chat_template: false}]\n"
            "train: {max_steps: 1, batch_size: 1, micro_batch_size: 1}\n"
            f"loss:\n  terms:\n    - {{criterion: {name}}}\n"
        )
        recipe_file.write_text(recipe_content, encoding="utf-8")

    print(f"Created scaffold for [{kind}] '{name}' (auto-discovered: no registration list to edit).")
    return 0

