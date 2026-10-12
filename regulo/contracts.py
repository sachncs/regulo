"""Typed contracts shared by the public training API.

The numerical kernels remain deliberately small, while these immutable
containers make the lifecycle and state boundaries explicit for callers.
"""

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

ArrayGroups: TypeAlias = dict[str, list[np.ndarray]]


class ReguloError(Exception):
    """Base class for errors raised by the public regulo API."""


class ValidationError(ValueError, ReguloError):
    """Raised when user-provided data or configuration is invalid."""


class SerializationError(ValueError, ReguloError):
    """Raised when a model checkpoint is missing, corrupt, or incompatible."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Immutable training policy used to describe a run."""

    batch: int = 32
    epochs: int = 500
    earlystop: bool = False
    patience: int = 10

    def __post_init__(self) -> None:
        if self.batch <= 0:
            raise ValidationError("batch must be positive.")
        if self.epochs <= 0:
            raise ValidationError("epochs must be positive.")
        if self.patience <= 0:
            raise ValidationError("patience must be positive.")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Immutable summary returned by :meth:`regulo.Runner.fit`."""

    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch: int | None
    stopped_early: bool
    stop_reason: str

    @property
    def epochs_ran(self) -> int:
        """Number of epochs completed by the run."""
        return len(self.train_loss)


@dataclass(slots=True)
class Checkpoint:
    """In-memory checkpoint containing model and optimizer state."""

    parameters: ArrayGroups
    optimizer: dict[str, object]
    epoch: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
