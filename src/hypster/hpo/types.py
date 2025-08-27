from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass(frozen=True)
class HpoInt:
    step: int | None = None
    scale: Literal["linear", "log"] = "linear"
    base: float = 10.0
    include_max: bool = True


@dataclass(frozen=True)
class HpoFloat:
    step: float | None = None
    scale: Literal["linear", "log"] = "linear"
    base: float = 10.0
    distribution: Literal["uniform", "loguniform", "normal", "lognormal"] | None = None
    center: float | None = None
    spread: float | None = None


@dataclass(frozen=True)
class HpoCategorical:
    ordered: bool = False
    weights: Sequence[float] | None = None
