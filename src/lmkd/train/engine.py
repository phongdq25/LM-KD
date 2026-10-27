"""Execution engine: device placement, backward, AMP scaling, distributed reductions (DESIGN §11.7).

``local`` = single process (CPU / 1 GPU), fully working (incl. fp16 GradScaler).
``accelerate`` = DeepSpeed ZeRO-1/2/3 (accelerate only initializes rank/device), task T05.

Contract used by StepStrategy (do not bypass):
  * the student is run through ``rt.student_module`` = ``engine.wrap_trainable(HiddenForward(...))``
    so that the engine wrapper's ``forward`` is always on the call path (K1);
  * ``backward(loss)``      -> scales the loss when a GradScaler is active;
  * ``unscale(optimizers)`` -> call once before reading/clipping grads;
  * ``step_optimizers(optimizers) -> bool`` -> False when the scaler skipped the step (inf/NaN),
    in which case schedulers must NOT be stepped (K2);
  * ``handles_clipping``    -> True when the backend clips grads itself (DeepSpeed), so the builder
    does not add a ClipNorm transform (K5).

Checkpoint API (T05, ``train/checkpointing.py``): ``save_model``, ``iter_full_state_dict``,
``gathered``, ``save_optimizers`` / ``load_optimizers`` (built on ``optimizer_state_dict``),
``all_gather_object``. Every one of them is a collective under DeepSpeed: call on all ranks.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import torch
import torch.nn as nn

from lmkd.compat import ds_api
from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import ENGINES


class Engine(Plugin):
    """Execution backend (device, precision, backward, optimizer stepping, collectives, checkpoint IO). Two
    implementations: local and DeepSpeed.
    """
    world_size: int = 1
    rank: int = 0
    handles_clipping: bool = False
    prepare_before_optim: bool = False

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.scaler: Any = None   # torch.amp.GradScaler when fp16

    @property
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    # ---------------------------------------------------------------- wrapping
    def wrap_trainable(self, module: nn.Module) -> nn.Module:
        """Wrap the trainable forward module. Local: identity; DeepSpeed prepares it later."""
        return module

    def prepare(self, student_module: nn.Module, modules: nn.Module, optimizers: dict[str, Any]) -> tuple:
        return student_module, modules, optimizers

    def place_frozen(self, model: nn.Module, placement: str) -> nn.Module:
        raise NotImplementedError

    # ---------------------------------------------------------------- backward / step
    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale(self, optimizers: dict[str, Any]) -> None:
        if self.scaler is not None:
            for opt in optimizers.values():
                self.scaler.unscale_(opt)

    def step_optimizers(self, optimizers: dict[str, Any]) -> bool:
        """Step every optimizer. Returns False if the step was skipped (fp16 overflow)."""
        if self.scaler is None:
            for opt in optimizers.values():
                opt.step()
            return True
        before = self.scaler.get_scale()
        for opt in optimizers.values():
            self.scaler.step(opt)
        self.scaler.update()
        return self.scaler.get_scale() >= before

    def no_sync(self, module: nn.Module, enabled: bool):
        return contextlib.nullcontext()

    def all_reduce_sum(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def autocast(self):
        return contextlib.nullcontext()

    def clip_grad_norm(self, params, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)

    def get_global_grad_norm(self) -> float | None:
        return None

    def barrier(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {"scaler": self.scaler.state_dict()} if self.scaler is not None else {}

    def load_state_dict(self, state: dict) -> None:
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

    # ---------------------------------------------------------------- collectives / checkpoint (T05)
    is_partitioned: bool = False   # True when parameters are sharded across ranks (ZeRO-3)

    def all_gather_object(self, obj: Any) -> list:
        return [obj]

    def gathered(self, params: list[torch.nn.Parameter], modifier_rank: int | None = None):
        """Context in which ``params`` hold full tensors (ZeRO-3 gather). Local: no-op."""
        return contextlib.nullcontext()

    def iter_full_state_dict(self, model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
        """Stream (name, full tensor) pairs; rank 0 always receives full tensors (T08 weight sync)."""
        yield from model.state_dict().items()

    def load_model_state_dict(self, model: nn.Module, state: dict) -> Any:
        """Collective model-state loading; ZeRO-3 broadcasts rank-zero modifications."""
        with self.gathered(list(model.parameters()), modifier_rank=0), torch.no_grad():
            if self.is_main or not self.is_partitioned:
                return model.load_state_dict(state, strict=False)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def save_model(self, model: nn.Module, path: Path, tokenizer: Any) -> None:
        """HF format (adapter-only for PEFT) + tokenizer, written by rank 0."""
        if self.is_main and hasattr(model, "save_pretrained"):
            model.save_pretrained(path)  # type: ignore[operator]
            if tokenizer is not None:
                tokenizer.save_pretrained(path)

    def optimizer_state_dict(self, opt: Any) -> dict:
        return opt.state_dict()

    def load_optimizer_state_dict(self, opt: Any, sd: dict) -> None:
        opt.load_state_dict(sd)

    def save_optimizers(self, optimizers: dict[str, Any], path: Path) -> None:
        """``path/<name>.pt`` per optimizer (rank 0)."""
        if self.is_main:
            Path(path).mkdir(parents=True, exist_ok=True)
            for name, opt in optimizers.items():
                torch.save(self.optimizer_state_dict(opt), Path(path) / f"{name}.pt")

    def load_optimizers(self, optimizers: dict[str, Any], path: Path) -> None:
        for name, opt in optimizers.items():
            sd = torch.load(Path(path) / f"{name}.pt", map_location="cpu", weights_only=False)
            self.load_optimizer_state_dict(opt, sd)


class MasterWeightOptimizer(torch.optim.Optimizer):
    """Wraps client optimizer(s) to maintain FP32 master weights for BF16/FP16 parameters.

    In pure BF16, optimizer updates can stall due to 7-bit mantissa truncation.
    This wrapper keeps FP32 master copies of parameters, applies optimizer updates in FP32,
    and copies results back to the model's low-precision parameters.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.param_map: dict[torch.nn.Parameter, torch.nn.Parameter] = {}

        master_groups = []
        for g in optimizer.param_groups:
            new_g = {k: v for k, v in g.items() if k != "params"}
            new_params = []
            for p in g["params"]:
                if p.dtype in (torch.bfloat16, torch.float16):
                    master_p = nn.Parameter(p.detach().clone().float(), requires_grad=p.requires_grad)
                    self.param_map[p] = master_p
                    new_params.append(master_p)
                else:
                    new_params.append(p)
            new_g["params"] = new_params
            master_groups.append(new_g)

        optimizer.param_groups = master_groups
        self.param_groups = master_groups
        self.defaults = optimizer.defaults
        self.state = optimizer.state

    def step(self, closure=None):
        for p, master_p in self.param_map.items():
            if p.grad is not None:
                master_p.grad = p.grad.float()
            else:
                master_p.grad = None
        loss = self.optimizer.step(closure=closure)
        for p, master_p in self.param_map.items():
            p.data.copy_(master_p.data)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p, master_p in self.param_map.items():
            if set_to_none:
                p.grad = None
                master_p.grad = None
            else:
                if p.grad is not None:
                    p.grad.zero_()
                if master_p.grad is not None:
                    master_p.grad.zero_()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)
        for p, master_p in self.param_map.items():
            master_p.data.copy_(p.data.float())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)


@ENGINES.register("local")
class LocalEngine(Engine):
    """CPU or single-GPU engine with optional bf16/fp16 autocast and GradScaler."""
    class Config(BaseConfig):
        device: str = "auto"            # auto | cpu | cuda | cuda:N | mps
        mixed_precision: str = "bf16"   # no | bf16 | fp16 (fp16 => GradScaler on CUDA)
        master_weights: bool = True     # maintain FP32 master weights for BF16/FP16 parameters

    def __init__(self, cfg: LocalEngine.Config):
        super().__init__(cfg)
        d = cfg.device
        if d == "auto":
            d = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(d)
        if self._device.type == "cuda":
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if cfg.mixed_precision == "fp16":
                self.scaler = torch.amp.GradScaler("cuda")

    @property
    def device(self) -> torch.device:
        return self._device

    def prepare(self, student_module: nn.Module, modules: nn.Module, optimizers: dict[str, Any]) -> tuple:
        if self.cfg.master_weights:
            wrapped = {}
            for name, opt in optimizers.items():
                has_low_prec = any(
                    p.dtype in (torch.bfloat16, torch.float16)
                    for g in opt.param_groups
                    for p in g["params"]
                )
                if has_low_prec and not isinstance(opt, MasterWeightOptimizer):
                    wrapped[name] = MasterWeightOptimizer(opt)
                else:
                    wrapped[name] = opt
            optimizers = wrapped
        return student_module, modules, optimizers

    def place_frozen(self, model: nn.Module, placement: str) -> nn.Module:
        """Place a frozen model (teacher).

        Note: 'cpu_offload' uses accelerate module offloading. It is much slower than
        device placement because modules are transferred on every forward pass, and
        should only be used when the teacher model does not fit in VRAM.
        """
        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            return model
        if placement == "cpu":
            return model.to("cpu")
        if placement == "cpu_offload":
            from accelerate import cpu_offload

            cpu_offload(model, execution_device=self._device, offload_buffers=True)
            return model
        if placement.startswith("device:"):
            return model.to(placement.split(":", 1)[1])
        return model.to(self._device)

    def autocast(self):
        if self._device.type == "cuda" and self.cfg.mixed_precision in ("bf16", "fp16"):
            dt = torch.bfloat16 if self.cfg.mixed_precision == "bf16" else torch.float16
            return torch.autocast("cuda", dtype=dt)
        return contextlib.nullcontext()


# ============================================================================ DeepSpeed (T05)
class TrainableBundle(nn.Module):
    """Everything DeepSpeed optimizes: the student forward module + aux modules (projectors...).

    ``forward(**kw)`` runs the student only; aux modules and the (tied) lm_head are called later by
    criteria / LogitsView as ordinary submodule calls, so DeepSpeed's module hooks (ZeRO-3 gather)
    and grad hooks (ZeRO-1/2 reduction) still manage them. The teacher is NEVER inside the bundle.
    """

    def __init__(self, student: nn.Module, modules: nn.Module):
        super().__init__()
        self.student = student
        self.modules_ = modules

    def forward(self, **kw):
        return self.student(**kw)


class MultiOptimizer(torch.optim.Optimizer):
    """Combines multiple client optimizers (e.g. Muon for 2D weights + AdamW for 1D weights)
    into a single ``torch.optim.Optimizer`` interface for DeepSpeed.
    """

    def __init__(self, optimizers: dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers
        all_groups = []
        for opt in optimizers.values():
            all_groups.extend(opt.param_groups)
        # DeepSpeed inspects defaults and param_groups
        self.defaults = next(iter(optimizers.values())).defaults if optimizers else {}
        self.param_groups = all_groups
        self.state: Any = {}
        for opt in optimizers.values():
            self.state.update(opt.state)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for opt in self.optimizers.values():
            opt.step()
        return loss

    def state_dict(self) -> dict:
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        for name, opt in self.optimizers.items():
            if name in state_dict:
                opt.load_state_dict(state_dict[name])


class DeepSpeedForward(nn.Module):
    """``rt.student_module`` under DeepSpeed: every student forward goes through ``ds_engine(**kw)``
    (K1). Holds the engine without registering it as a submodule (no double parameter listing)."""

    ds_engine: Any   # set with object.__setattr__ (bypasses nn.Module registration)

    def __init__(self, ds_engine: Any):
        super().__init__()
        object.__setattr__(self, "ds_engine", ds_engine)

    def forward(self, **kw):
        return self.ds_engine(**kw)


class DeepSpeedOptimizer(torch.optim.Optimizer):
    """``rt.optimizers[name]`` under DeepSpeed.

    LR schedulers (our LambdaLR, never passed to DS) need a ``torch.optim.Optimizer``, which the ZeRO
    optimizer is not; it shares ``param_groups`` with the client optimizer, so scheduling the client
    groups is what DeepSpeed reads at ``step``. ``zero_grad`` / ``state_dict`` go to the ZeRO
    optimizer (it owns the gradient partitions). ``step`` is ``engine.step_optimizers``' job.
    """

    client: torch.optim.Optimizer
    _ds: Any

    def __init__(self, ds_engine: Any, client: torch.optim.Optimizer):
        # no Optimizer.__init__: param_groups/state belong to the client optimizer
        object.__setattr__(self, "_ds", ds_engine)
        object.__setattr__(self, "client", client)
        self.defaults = client.defaults
        self._opt_called = False

    @property
    def param_groups(self):
        return self.client.param_groups

    @param_groups.setter
    def param_groups(self, v):
        self.client.param_groups = v

    @property
    def state(self):
        return self.client.state

    @state.setter
    def state(self, v):
        self.client.state = v

    def step(self, closure=None):     # noqa: D401
        raise RuntimeError("under DeepSpeed the optimizer is stepped by engine.step_optimizers()")

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._ds.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return self._ds.optimizer.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self._ds.optimizer.load_state_dict(sd)

    def __repr__(self) -> str:
        return f"DeepSpeedOptimizer({type(self._ds.optimizer).__name__})"


_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_ds_config(spec: str | dict | None) -> dict:
    """DeepSpeed config from a dict, a JSON path (cwd-relative, then repo-relative) or None (= {})."""
    import copy
    import json

    if spec is None:
        return {}
    if isinstance(spec, dict):
        return copy.deepcopy(spec)
    p = Path(spec)
    if not p.is_absolute() and not p.exists() and (_REPO_ROOT / p).exists():
        p = _REPO_ROOT / p
    return json.loads(p.read_text())


@ENGINES.register("accelerate", source="white-box-distillation/same-tokenizer/finetune.py:557-570")
class AccelerateEngine(Engine):
    """DeepSpeed ZeRO-1/2/3 data-parallel engine; ``accelerate.PartialState`` only sets up rank/device.

    DeepSpeed owns backward (loss scaling), gradient reduction over the whole TrainableBundle
    (student + head + aux modules), clipping, precision and the optimizer step. The DS config is
    patched like finetune.py:557-570 (micro-batch, GA, clipping, precision) but the teacher is kept
    OUT of the DS engine (unlike IMPACT distillation.py:624-630, which casts/replicates it).

    ``batch_size`` / ``micro_batch_size`` / ``gradient_clipping`` mirror ``train.batch_size``,
    ``train.micro_batch_size`` and ``grad.clip_norm``: the engine is built before the runtime, so
    the base recipes ``recipes/_base_/engine_zero*.yaml`` fill them with ``${train.*}`` / ``${grad.*}``.
    When left ``None`` the values already in ``deepspeed_config`` are used.
    """

    handles_clipping = True

    class Config(BaseConfig):
        backend: Literal["deepspeed"] = "deepspeed"
        mixed_precision: str = "bf16"       # no | bf16 | fp16
        deepspeed_config: str | dict | None = None   # path or dict; ZeRO stage, offload...
        timeout_minutes: int = 60
        batch_size: int | None = None              # = train.batch_size (global per optimizer step)
        micro_batch_size: int | None = None        # = train.micro_batch_size (per device)
        gradient_clipping: float | None = None     # = grad.clip_norm (None/0 = no clipping)

    def __init__(self, cfg: AccelerateEngine.Config):
        super().__init__(cfg)
        if cfg.mixed_precision not in ("no", "bf16", "fp16"):
            raise ValueError(f"mixed_precision must be no|bf16|fp16, got {cfg.mixed_precision!r}")
        # Initializes the process group (NCCL on GPU, gloo on CPU) from torchrun env vars.
        import torch.distributed as dist
        from accelerate import PartialState

        st = PartialState(timeout=timedelta(minutes=cfg.timeout_minutes))
        if dist.is_available() and dist.is_initialized():   # PartialState under-reports CPU/gloo runs
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        else:
            self.rank, self.world_size = st.process_index, st.num_processes
        self._device = st.device
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        self.ds_engine: Any = None
        self.ds_config: dict = {}
        self.n_micro = 1
        self._frozen: list[nn.Module] = []

    def setup(self, rt: Any) -> None:
        """Read train/grad batch size and clipping from rt.config if not explicitly set."""
        if hasattr(rt, "config"):
            if self.cfg.batch_size is None and hasattr(rt.config, "train"):
                self.cfg.batch_size = rt.config.train.batch_size
            if self.cfg.micro_batch_size is None and hasattr(rt.config, "train"):
                self.cfg.micro_batch_size = rt.config.train.micro_batch_size
            if self.cfg.gradient_clipping is None and hasattr(rt.config, "grad"):
                self.cfg.gradient_clipping = rt.config.grad.clip_norm

    def get_global_grad_norm(self) -> float | None:
        if self.ds_engine is not None and hasattr(self.ds_engine, "get_global_grad_norm"):
            try:
                norm = self.ds_engine.get_global_grad_norm()
                if norm is not None:
                    return float(norm)
            except Exception:
                pass
        return None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def zero_stage(self) -> int:
        return int(self.ds_config.get("zero_optimization", {}).get("stage", 0))

    @property
    def is_partitioned(self) -> bool:      # type: ignore[override]
        return self.zero_stage == 3

    # ---------------------------------------------------------------- placement / wrapping
    def place_frozen(self, model: nn.Module, placement: str) -> nn.Module:
        """Place a frozen model (teacher) under Accelerate/DeepSpeed."""
        if placement == "shard":
            raise NotImplementedError("teacher placement 'shard' (ZeRO-3 inference) is task T26")
        self._frozen.append(model)
        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            return model
        if placement == "cpu":
            return model.to("cpu")
        if placement == "cpu_offload":
            from accelerate import cpu_offload

            cpu_offload(model, execution_device=self._device, offload_buffers=True)
            return model
        if placement.startswith("device:"):
            return model.to(placement.split(":", 1)[1])
        return model.to(self._device)

    def wrap_trainable(self, module: nn.Module) -> nn.Module:
        return module                      # DeepSpeed wraps the whole bundle in prepare()

    def build_ds_config(self, n_params_groups: int = 1) -> dict:
        c = self.cfg
        ds = load_ds_config(c.deepspeed_config)

        def given(key: str):               # HF-style "auto" = "fill from the recipe"
            v = ds.get(key)
            return None if v == "auto" else v

        mbs = c.micro_batch_size or given("train_micro_batch_size_per_gpu")
        bs = c.batch_size or given("train_batch_size")
        if mbs is None or bs is None:
            raise ValueError("accelerate/deepspeed needs batch_size and micro_batch_size (use "
                             "recipes/_base_/engine_zero*.yaml, which set them from ${train.*})")
        if bs % (mbs * self.world_size):
            raise ValueError(f"batch_size {bs} not divisible by micro_batch_size {mbs} x world_size {self.world_size}")
        ga = bs // (mbs * self.world_size)
        if int(ds.get("zero_optimization", {}).get("stage", 0)) == 0 and c.mixed_precision != "no":
            # Stage 0 + bf16/fp16 wraps the client optimizer in FP16_UnfusedOptimizer (DeepSpeed 0.19.8):
            # its __init__ runs one zero-grad optimizer.step() (AdamW step count +1 -> wrong bias
            # correction, one extra weight decay), and for fp16 engine.backward never multiplies the
            # loss by the loss scale while step() still divides by it (grads / 2^16, no overflow check).
            raise ValueError("DeepSpeed ZeRO stage 0 with mixed_precision bf16/fp16 trains incorrectly "
                             "(DeepSpeed 0.19.8 FP16_UnfusedOptimizer); use zero stage >= 1 or mixed_precision: no")
        ds["train_micro_batch_size_per_gpu"] = mbs
        ds["gradient_accumulation_steps"] = ga
        ds["train_batch_size"] = bs
        if c.gradient_clipping is not None or given("gradient_clipping") is None:
            ds["gradient_clipping"] = float(c.gradient_clipping or 0.0)
        ds["bf16"] = {**ds.get("bf16", {}), "enabled": c.mixed_precision == "bf16"}
        ds["fp16"] = {**ds.get("fp16", {}), "enabled": c.mixed_precision == "fp16"}
        ds.setdefault("zero_allow_untested_optimizer", True)   # client optimizers (AdamW, Muon...)
        if ds.get("zero_optimization", {}).get("offload_optimizer", {}).get("device", "none") != "none":
            ds.setdefault("zero_force_ds_cpu_optimizer", False)
        ds.pop("optimizer", None)          # our optimizer / scheduler, never DS-built ones
        ds.pop("scheduler", None)
        return ds

    def prepare(self, student_module: nn.Module, modules: nn.Module, optimizers: dict[str, Any]) -> tuple:
        if not optimizers:
            raise ValueError("no optimizers configured")

        if len(optimizers) == 1:
            name, client = next(iter(optimizers.items()))
            wrapped_client = client
        else:
            name = "multi"
            wrapped_client = MultiOptimizer(optimizers)

        bundle = TrainableBundle(student_module, modules)
        inside = {id(m) for m in bundle.modules()}
        assert all(id(t) not in inside for t in self._frozen), "teacher must not be inside the DS engine"
        self.ds_config = self.build_ds_config()
        self.n_micro = self.ds_config["gradient_accumulation_steps"]
        ds_engine = ds_api.initialize(model=bundle, optimizer=wrapped_client, config=self.ds_config)
        self.ds_engine = ds_engine

        # Build optimizer map
        if len(optimizers) == 1:
            ret_opts = {name: DeepSpeedOptimizer(ds_engine, client)}
        else:
            # Map each original client optimizer to its DeepSpeed wrapper sharing param_groups
            ret_opts = {n: DeepSpeedOptimizer(ds_engine, opt) for n, opt in optimizers.items()}
        return DeepSpeedForward(ds_engine), bundle.modules_, ret_opts

    # ---------------------------------------------------------------- backward / step
    def no_sync(self, module: nn.Module, enabled: bool):
        @contextlib.contextmanager
        def ctx():
            ds_api.set_accumulation_boundary(self.ds_engine, not enabled)
            yield
        return ctx()

    def backward(self, loss: torch.Tensor) -> None:
        # ds_engine.backward divides the loss by gradient_accumulation_steps (scale_wrt_gas), but our
        # loss is already normalized by the GLOBAL Z (all micro-batches x ranks, DESIGN §6.2):
        # multiply by n_micro to undo it. (x world_size for the rank average is done in accumulate.)
        self.ds_engine.backward(loss * self.n_micro)

    def unscale(self, optimizers: dict[str, Any]) -> None:
        return None                        # DeepSpeed manages precision of the whole bundle

    def step_optimizers(self, optimizers: dict[str, Any]) -> bool:
        before = ds_api.skipped_steps(self.ds_engine)
        self.ds_engine.step()
        for opt in optimizers.values():
            opt._opt_called = True         # LambdaLR's "optimizer.step() was called" bookkeeping
        return ds_api.step_was_applied(self.ds_engine, before)

    def autocast(self):
        # DS casts the bundle to bf16/fp16; autocast keeps mixed-dtype ops (fp32 teacher features,
        # fp32 buffers in criteria) valid.
        if self._device.type == "cuda" and self.cfg.mixed_precision in ("bf16", "fp16"):
            dt = torch.bfloat16 if self.cfg.mixed_precision == "bf16" else torch.float16
            return torch.autocast("cuda", dtype=dt)
        return contextlib.nullcontext()

    def clip_grad_norm(self, params, max_norm: float) -> torch.Tensor:
        raise RuntimeError("DeepSpeed clips gradients itself (gradient_clipping); do not add ClipNorm")

    # ---------------------------------------------------------------- collectives
    def all_reduce_sum(self, x: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        y = x.clone()
        dist.all_reduce(y)
        return y

    def all_gather_object(self, obj: Any) -> list:
        import torch.distributed as dist

        out: list = [None] * self.world_size
        dist.all_gather_object(out, obj)
        return out

    def barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier()

    # ---------------------------------------------------------------- checkpoint
    def gathered(self, params: list[torch.nn.Parameter], modifier_rank: int | None = None):
        if not self.is_partitioned:
            return contextlib.nullcontext()
        return ds_api.gathered_parameters(params, modifier_rank)

    def iter_full_state_dict(self, model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
        """Collective: every rank must iterate to the end. Same keys as ``model.state_dict()``
        (tied weights included, non-persistent buffers excluded); one parameter gathered at a time."""
        if not self.is_partitioned:
            yield from model.state_dict().items()
            return
        for n, v in model.state_dict(keep_vars=True).items():
            if ds_api.is_partitioned_param(v):             # partitioned ZeRO-3 parameter
                with self.gathered([v]):
                    t = v.detach().to("cpu", copy=True)
            else:
                t = v.detach().to("cpu", copy=True)
            yield n, t

    def save_model(self, model: nn.Module, path: Path, tokenizer: Any) -> None:
        if not self.is_partitioned:
            return super().save_model(model, path, tokenizer)
        sd = dict(self.iter_full_state_dict(model))    # tied duplicates are absent: HF re-ties them
        if self.is_main and hasattr(model, "save_pretrained"):
            model.save_pretrained(path, state_dict=sd)  # type: ignore[operator]
            if tokenizer is not None:
                tokenizer.save_pretrained(path)

    def save_optimizers(self, optimizers: dict[str, Any], path: Path) -> None:
        # Collective: each rank writes its ZeRO optimizer partition (and DS's own module copy).
        self.ds_engine.save_checkpoint(str(Path(path) / "ds"), tag="state", save_latest=False)

    def load_optimizers(self, optimizers: dict[str, Any], path: Path) -> None:
        load_path, _ = self.ds_engine.load_checkpoint(
            str(Path(path) / "ds"), tag="state", load_optimizer_states=True,
            load_lr_scheduler_states=False, load_module_strict=False)
        if load_path is None:
            raise FileNotFoundError(f"no DeepSpeed optimizer state in {Path(path) / 'ds'}")


# ============================================================================ FSDP2 (Native PyTorch 2.x Distributed Engine)
@ENGINES.register("fsdp2")
class FSDP2Engine(Engine):
    """Native PyTorch FullyShardedDataParallel (FSDP2 / per-parameter sharded) execution engine.

    Uses `torch.distributed.fsdp.fully_shard` and standard PyTorch DDP collectives.
    Optimizers bind after sharding. Standard first-order steps and full checkpoints
    are supported; SAM/PCGrad remain restricted by their step-strategy guards.
    """
    prepare_before_optim = True
    is_partitioned = True

    class Config(BaseConfig):
        mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
        gradient_clipping: float | None = None
        reshard_after_forward: bool = False

    def __init__(self, cfg: FSDP2Engine.Config | None = None):
        super().__init__(cfg or self.Config())
        import os

        import torch.distributed as dist

        if not dist.is_initialized():
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.cfg.mixed_precision == "fp16")) if self._device.type == "cuda" else None

    @property
    def device(self) -> torch.device:
        return self._device

    def setup(self, rt: Any) -> None:
        """Fail before training on unsupported adapter checkpoint semantics."""
        if hasattr(rt.student_model, "peft_config"):
            raise ValueError("FSDP2 PEFT adapter-only checkpoint/resume is unsupported; use local or DeepSpeed")
        if self.cfg.gradient_clipping is not None:
            rt.config.grad.clip_norm = self.cfg.gradient_clipping
        if hasattr(rt, "config") and hasattr(rt.config, "train"):
            bs = rt.config.train.batch_size
            mbs = rt.config.train.micro_batch_size
            if bs % (mbs * self.world_size) != 0:
                raise ValueError(f"batch_size {bs} not divisible by micro_batch_size {mbs} x world_size {self.world_size}")

    def wrap_trainable(self, module: nn.Module) -> nn.Module:
        return module.to(self.device)

    def prepare(self, student_module: nn.Module, modules: nn.Module, optimizers: dict[str, Any]) -> tuple:
        from lmkd.compat import fsdp_api

        if self.cfg.reshard_after_forward:
            raise ValueError("FSDP2 lazy logits/aux need reshard_after_forward=false")
        if any(opt.state for opt in optimizers.values()):
            raise ValueError("FSDP2 prepare needs fresh optimizers; load their state after sharding")
        bundle = TrainableBundle(student_module, modules).to(self.device)
        old_names = {id(p): n for n, p in bundle.named_parameters()}
        for child in bundle.modules():
            if isinstance(child, nn.Linear):
                child._lmkd_fsdp_managed = True  # type: ignore[assignment]
        fsdp_api.shard(bundle)
        current = dict(bundle.named_parameters())
        for opt in optimizers.values():
            for group in opt.param_groups:
                group["params"] = [current[old_names[id(p)]] for p in group["params"]]
        self._bundle = bundle
        return bundle, bundle.modules_, optimizers

    def place_frozen(self, model: nn.Module, placement: str) -> nn.Module:
        return model.to(self.device).eval()

    def autocast(self):
        if self._device.type == "cuda" and self.cfg.mixed_precision in ("bf16", "fp16"):
            dt = torch.bfloat16 if self.cfg.mixed_precision == "bf16" else torch.float16
            return torch.autocast("cuda", dtype=dt)
        return contextlib.nullcontext()

    def all_reduce_sum(self, x: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return x
        import torch.distributed as dist

        t = x.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    def no_sync(self, module: nn.Module, enabled: bool):
        if not enabled:
            return contextlib.nullcontext()
        return self._gradient_sync(module)

    @contextlib.contextmanager
    def _gradient_sync(self, module):
        module.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            module.set_requires_gradient_sync(True)

    def clip_grad_norm(self, params, max_norm: float) -> torch.Tensor:
        from lmkd.compat.fsdp_api import full_tensor

        return full_tensor(torch.nn.utils.clip_grad_norm_(params, max_norm))

    def iter_full_state_dict(self, model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
        from lmkd.compat.fsdp_api import full_model_state

        yield from full_model_state(model).items()

    def load_model_state_dict(self, model: nn.Module, state: dict) -> Any:
        from lmkd.compat.fsdp_api import load_model_state

        return load_model_state(model, state)

    def save_model(self, model: nn.Module, path: Path, tokenizer: Any) -> None:
        state = dict(self.iter_full_state_dict(model))
        if self.is_main:
            model.save_pretrained(path, state_dict=state)  # type: ignore[operator]
            if tokenizer is not None:
                tokenizer.save_pretrained(path)

    def optimizer_state_dict(self, opt: Any) -> dict:
        from lmkd.compat.fsdp_api import full_optimizer_state

        return full_optimizer_state(self._bundle, opt)

    def load_optimizer_state_dict(self, opt: Any, sd: dict) -> None:
        from lmkd.compat.fsdp_api import load_optimizer_state

        load_optimizer_state(self._bundle, opt, sd)

    def save_optimizers(self, optimizers: dict[str, Any], path: Path) -> None:
        for name, opt in optimizers.items():
            state = self.optimizer_state_dict(opt)
            if self.is_main:
                Path(path).mkdir(parents=True, exist_ok=True)
                torch.save(state, Path(path) / f"{name}.pt")

    def all_gather_object(self, obj: Any) -> list:
        if self.world_size == 1:
            return [obj]
        import torch.distributed as dist

        out: list = [None] * self.world_size
        dist.all_gather_object(out, obj)
        return out

    def barrier(self) -> None:
        if self.world_size > 1:
            import torch.distributed as dist

            dist.barrier()



