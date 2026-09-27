"""Versioned, validated settings shared by training and inference."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    max_candidates: int = 50
    max_block_size: int = 500
    baseline_threshold: float = 0.86
    cv_folds: int = 5
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1

    def __post_init__(self):
        for key in ("max_candidates", "max_block_size", "cv_folds", "n_estimators", "max_depth"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        for key in ("baseline_threshold", "learning_rate"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32)")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**data)
