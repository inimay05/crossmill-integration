from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from crossmill.config import (
    ABSTRACT_DIM,
    BIAS_VECTOR_DIM,
    DEFAULT_MEMORY_CONFIG,
)

_REQUIRED_DIFFICULTY_KEYS = {'easy', 'medium', 'hard'}


class MemoryConfig(BaseModel):
    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'mode': 'cross',
                    'transfer_direction': 'bidirectional',
                    'top_k': 3,
                    'min_confidence': 0.1,
                    'promotion_thresholds': {'easy': 3, 'medium': 4, 'hard': 5},
                    'initial_min_confidence': 0.4,
                    'confidence_warmup_steps': 50000,
                }
            ]
        }
    }

    mode: Literal['none', 'local', 'cross'] = DEFAULT_MEMORY_CONFIG['mode']
    transfer_direction: Literal['steel_to_food', 'food_to_steel', 'bidirectional'] = (
        DEFAULT_MEMORY_CONFIG['transfer_direction']
    )
    top_k: int = Field(DEFAULT_MEMORY_CONFIG['top_k'], ge=1, le=10)
    min_confidence: float = Field(DEFAULT_MEMORY_CONFIG['min_confidence'], ge=0.0, le=1.0)
    promotion_thresholds: dict[str, int] = Field(
        default_factory=lambda: dict(DEFAULT_MEMORY_CONFIG['promotion_thresholds'])
    )
    initial_min_confidence: float = Field(
        DEFAULT_MEMORY_CONFIG['initial_min_confidence'], ge=0.0, le=1.0
    )
    confidence_warmup_steps: int = Field(
        DEFAULT_MEMORY_CONFIG['confidence_warmup_steps'], ge=0
    )

    @model_validator(mode='after')
    def _validate_memory_config(self) -> MemoryConfig:
        if self.initial_min_confidence < self.min_confidence:
            raise ValueError(
                f'initial_min_confidence ({self.initial_min_confidence}) must be '
                f'>= min_confidence ({self.min_confidence})'
            )
        missing = _REQUIRED_DIFFICULTY_KEYS - self.promotion_thresholds.keys()
        if missing:
            raise ValueError(
                f'promotion_thresholds missing required keys: {missing}'
            )
        below_minimum = {
            k: v for k, v in self.promotion_thresholds.items() if v < 2
        }
        if below_minimum:
            raise ValueError(
                f'promotion_thresholds values must be >= 2, got: {below_minimum}'
            )
        return self


class MemoryRecord(BaseModel):
    env_id: str
    abstract_state: list[float]
    action_pattern: Literal[
        'gradual_ramp',
        'hold_steady',
        'rapid_correction',
        'cautious_explore',
        'emergency_response',
    ]
    outcome_reward: float
    is_negative: bool = False
    safety_regime: Literal['strict', 'moderate', 'permissive']
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    access_count: int = Field(0, ge=0)
    task_difficulty: Literal['easy', 'medium', 'hard']
    provenance: list[str] = Field(default_factory=list)
    timestamp: int = Field(0, ge=0)

    @model_validator(mode='after')
    def _validate_abstract_state(self) -> MemoryRecord:
        if len(self.abstract_state) != ABSTRACT_DIM:
            raise ValueError(
                f'abstract_state must have length {ABSTRACT_DIM}, '
                f'got {len(self.abstract_state)}'
            )
        out_of_range = [v for v in self.abstract_state if not (0.0 <= v <= 1.0)]
        if out_of_range:
            raise ValueError(
                f'all abstract_state values must be in [0, 1]; '
                f'found {len(out_of_range)} out-of-range value(s): {out_of_range}'
            )
        return self


class AugmentedObservation(BaseModel):
    env_name: str
    raw_obs_vector: list[float]
    bias_vector: list[float] = Field(
        default_factory=lambda: [0.0] * BIAS_VECTOR_DIM
    )
    retrieved_memories: list[MemoryRecord] = Field(default_factory=list)

    @computed_field
    @property
    def augmented_vector(self) -> list[float]:
        return self.raw_obs_vector + self.bias_vector


class TransferResult(BaseModel):
    bias_vector: list[float]
    retrieved_memories: list[MemoryRecord]
    retrieval_scores: list[float]
    gate_active: bool


if __name__ == '__main__':
    cfg = MemoryConfig()
    print('MemoryConfig defaults:', cfg.model_dump())
    rec = MemoryRecord(
        env_id='megaforge', abstract_state=[0.5]*8,
        action_pattern='gradual_ramp', outcome_reward=0.3,
        safety_regime='strict', task_difficulty='easy')
    print('MemoryRecord example:', rec.model_dump())
    print('Schemas OK')
