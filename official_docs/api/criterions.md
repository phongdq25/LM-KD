# API Reference: Criterions & Losses

Technical reference for building custom loss functions and distillation objectives.

---

## 1. `lmkd.losses.base.Criterion`

Base class for all distillation loss implementations (`torch.nn.Module`).

### Method Signatures

```python
class Criterion(nn.Module):
    """The loss contract.

    A criterion computes unnormalized loss numerators and declares required features.
    It never shifts logits and never divides by batch size.
    """

    def features(self) -> FeatureRequest:
        """Declares what tensors (logits, hidden layers, attentions) the engine must extract."""
        return FeatureRequest()

    def forward(self, ctx: LossContext) -> LossOutput:
        """Computes loss terms and diagnostic metrics for one micro-batch."""
        raise NotImplementedError
```

---

## 2. `lmkd.losses.base.LossContext`

Context object delivered to `Criterion.forward()` containing all inputs and model activations:

```python
@dataclass
class LossContext:
    batch: Batch                       # The universal batch container
    student: SideOutput                # Student forward activations & hidden states
    student_views: list[SideOutput]    # Multi-view student states (for SimCSE/contrastive)
    teacher: TeacherOutput | None      # Teacher activations and signals (None if student-only)
    modules: ModuleRegistry            # Auxiliary learnable modules (e.g. projectors)
    state: TrainState                  # Training step, epoch, and progress state
    rt: Runtime                        # Read-only reference to the active runtime

    def student_logits(self) -> LogitsView:
        """Target-aligned LogitsView for the student model."""
        ...

    def supervised_mask(self, side: str = "student") -> torch.Tensor:
        """(labels != -100) * seq.loss_mask (if present), float [B, T]."""
        ...
```

---

## 3. `lmkd.losses.base.LossOutput` & `LossTerm`

The data structure returned by every `Criterion.forward()`.

### Class Definitions
```python
@dataclass
class LossTerm:
    value: torch.Tensor           # Unnormalized scalar numerator with grad
    reduction: Literal["token_sum", "seq_sum", "mean"]
    log_value: torch.Tensor | None = None  # Diagnostic value to log (if different)

@dataclass
class LossOutput:
    terms: dict[str, LossTerm]    # Named loss numerators to be optimized
    metrics: dict[str, torch.Tensor] = field(default_factory=dict) # Detached diagnostics
```

- `"token_sum"`: Divided globally by $Z_{\text{tok}}$ (total supervised tokens across all ranks and micro-batches).
- `"seq_sum"`: Divided globally by $Z_{\text{seq}}$ (total sequence count across the optimizer step).
- `"mean"`: Divided by micro-batch count (already normalized internally).

---

## 4. `lmkd.models.forward.FeatureRequest` & `SideRequest`

Informs the execution engine which activations must be retained during the forward pass:

```python
@dataclass
class SideRequest:
    logits: Literal["none", "view", "full"] = "none"
    hidden: LayerSpec = "none"                         # tuple of layer indices, "last", or "all"
    attentions: LayerSpec = "none"
    input_embeds: bool = False
    pooled: tuple[tuple[int, str], ...] = ()           # (layer, pooler_name)
    n_views: int = 1                                   # >1 for stochastic forward passes

@dataclass
class FeatureRequest:
    student: SideRequest = field(default_factory=SideRequest)
    teacher: SideRequest | None = None
    teacher_signal: Any | None = None                  # Offline or streamed teacher signals
```

---

## 5. `lmkd.models.logits_view.LogitsView`

Target-aligned, lazy access to token distributions without materializing huge `[Batch, Length, Vocab]` tensors in memory:

```python
class LogitsView:
    def full(self) -> torch.Tensor:
        """Materializes the full fp32 tensor [B, T, V] (target-aligned)."""
        ...

    def chunks(self, chunk_size: int = 1024) -> Iterator[LogitChunk]:
        """Yields streaming target-aligned slices [B, chunk, V] to conserve memory."""
        ...

    def target_logp(self, labels: torch.Tensor) -> torch.Tensor:
        """Target token log-probabilities without materializing full vocabulary."""
        ...

    def topk(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns top-k values and token indices along vocabulary dimension."""
        ...

    def with_temperature(self, t: float) -> LogitsView:
        """Returns a new LogitsView scaled by temperature."""
        ...
```

---

## 6. Functional Loss Decorator: `@lmkd.losses.loss`

For simple token-level or sequence-level objectives, you can define a criterion using the functional decorator:

```python
from lmkd.losses.loss_decorator import loss, TokenLoss

@loss("my_quick_loss", teacher=True, chunked=True)
def my_quick_loss(student, teacher, temperature: float = 1.0):
    p_t = torch.softmax(teacher.logits / temperature, dim=-1)
    p_s = torch.log_softmax(student.logits / temperature, dim=-1)
    kl = (p_t * (p_t.log() - p_s)).sum(dim=-1)
    return TokenLoss(values=kl * (temperature ** 2))
```

