from collections import deque, defaultdict
from crossmill.models import MemoryRecord, MemoryConfig, TransferResult
from crossmill.memory.config import (
    BIAS_VECTOR_DIM,
    BIAS_CLIP,
    EMA_ALPHA,
    CONFIDENCE_FLOOR,
    CONFIDENCE_CEILING,
    BASELINE_BUFFER_MAXLEN,
    PATTERN_BIAS,
    ENVIRONMENTS,
)


class TransferAdapter:
    """
    Builds bias vectors from retrieved memories and updates memory confidence
    using an exponential moving average on the episode-level reward delta.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._baseline: dict[str, deque] = {
            name: deque(maxlen=BASELINE_BUFFER_MAXLEN)
            for name in ENVIRONMENTS.keys()
        }
        self._episode_retrievals: dict[str, list[MemoryRecord]] = defaultdict(list)

    def build_bias_vector(self, records: list[MemoryRecord],
                          scores: list[float]) -> list[float]:
        """
        Aggregate the per-memory bias contributions into one 8-dim vector,
        then clip to [-BIAS_CLIP, +BIAS_CLIP].

        For each retrieved memory:
          sign = -1 if memory.is_negative else +1
          contribution = score * memory.confidence * sign * PATTERN_BIAS[label]
        Sum across all records, clip, return.

        If records is empty, return an 8-element zero vector.
        """
        if not records:
            return [0.0] * BIAS_VECTOR_DIM

        if len(records) != len(scores):
            raise ValueError(f"records ({len(records)}) and scores "
                             f"({len(scores)}) must have same length")

        bias = [0.0] * BIAS_VECTOR_DIM
        for rec, score in zip(records, scores):
            pattern_vec = PATTERN_BIAS.get(rec.action_pattern)
            if pattern_vec is None:
                continue
            sign = -1.0 if rec.is_negative else 1.0
            weight = score * rec.confidence * sign
            for i in range(BIAS_VECTOR_DIM):
                bias[i] += weight * pattern_vec[i]

        bias = [max(-BIAS_CLIP, min(BIAS_CLIP, v)) for v in bias]
        return bias

    def track_retrieval(self, env_name: str,
                        records: list[MemoryRecord]) -> None:
        """
        Mark these records as retrieved during the current episode on env_name.
        Called by CrossIndustryMemory.on_step.
        """
        self._episode_retrievals[env_name].extend(records)

    def update_confidence(self, env_name: str, episode_reward: float) -> dict:
        """
        Called at episode end. See class docstring for details.
        """
        baseline_buffer = self._baseline[env_name]
        if len(baseline_buffer) == 0:
            baseline = 0.0
        else:
            baseline = sum(baseline_buffer) / len(baseline_buffer)

        denom = max(abs(baseline), 0.01)
        signal = (episode_reward - baseline) / denom
        signal = max(-1.0, min(1.0, signal))

        retrieved = self._episode_retrievals[env_name]

        updated_count = 0
        seen_ids = set()
        for rec in retrieved:
            rec_id = id(rec)
            if rec_id in seen_ids:
                continue
            seen_ids.add(rec_id)
            old = rec.confidence
            new = (1.0 - EMA_ALPHA) * old + EMA_ALPHA * signal
            new = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, new))
            rec.confidence = new
            updated_count += 1

        baseline_buffer.append(episode_reward)
        self._episode_retrievals[env_name] = []

        return {
            'baseline': baseline,
            'signal': signal,
            'updated_count': updated_count,
            'skipped_count': 0,
        }

    def to_transfer_result(self, records: list[MemoryRecord],
                           scores: list[float],
                           gate_active: bool) -> TransferResult:
        """
        Convenience: build a TransferResult from a retrieval outcome.
        """
        bias = self.build_bias_vector(records, scores)
        return TransferResult(
            bias_vector=bias,
            retrieved_memories=list(records),
            retrieval_scores=list(scores),
            gate_active=gate_active,
        )


if __name__ == '__main__':
    from crossmill.memory.config import SAFETY_REGIME_BY_DIFFICULTY

    def make_rec(pattern, conf, is_neg=False, env='safenutri'):
        return MemoryRecord(
            env_id=env,
            abstract_state=[0.5] * 8,
            action_pattern=pattern,
            outcome_reward=0.2 if not is_neg else -0.8,
            is_negative=is_neg,
            safety_regime=SAFETY_REGIME_BY_DIFFICULTY['easy'],
            confidence=conf,
            task_difficulty='easy',
            provenance=['manual'],
        )

    cfg = MemoryConfig()
    adapter = TransferAdapter(cfg)

    rec1 = make_rec('gradual_ramp', 0.8)
    rec2 = make_rec('rapid_correction', 0.5)
    rec3 = make_rec('rapid_correction', 0.7, is_neg=True)

    # ---- Bias vector: [rec1, rec2] ----
    bias_a = adapter.build_bias_vector([rec1, rec2], [0.9, 0.6])
    assert len(bias_a) == 8, f"expected 8-dim, got {len(bias_a)}"
    assert all(-BIAS_CLIP <= v <= BIAS_CLIP for v in bias_a), \
        f"bias out of [-{BIAS_CLIP}, +{BIAS_CLIP}]: {bias_a}"
    print(f"Bias vector [rec1(+gradual, c=0.8, s=0.9), rec2(+rapid, c=0.5, s=0.6)]:")
    print(f"  {[f'{v:+.3f}' for v in bias_a]}")
    print(f"  length=8, all in [-0.5, +0.5] — OK")

    # ---- Sign inversion: [rec3] alone vs [rec2] alone ----
    bias_pos = adapter.build_bias_vector([rec2], [0.9])
    bias_neg = adapter.build_bias_vector([rec3], [0.9])
    print(f"\nSign inversion check (rapid_correction):")
    print(f"  rec2 (+positive, c=0.5): {[f'{v:+.3f}' for v in bias_pos]}")
    print(f"  rec3 (-negative, c=0.7): {[f'{v:+.3f}' for v in bias_neg]}")
    # Find dims where rec2 pattern is non-zero and confirm signs differ
    rapid_vec = PATTERN_BIAS['rapid_correction']
    for i, p in enumerate(rapid_vec):
        if p != 0.0:
            assert (bias_pos[i] > 0) != (bias_neg[i] > 0), \
                f"dim {i}: signs should differ; pos={bias_pos[i]} neg={bias_neg[i]}"
    print(f"  Signs are inverted on non-zero pattern dims — OK")

    # ---- Empty records ----
    empty = adapter.build_bias_vector([], [])
    assert empty == [0.0] * 8, f"empty expected [0]*8, got {empty}"
    print(f"\nEmpty records: {empty} — OK")

    # ---- EMA confidence update: upward ----
    adapter.track_retrieval('safenutri', [rec1, rec2])
    before1, before2 = rec1.confidence, rec2.confidence
    info = adapter.update_confidence('safenutri', episode_reward=2.0)
    assert info['baseline'] == 0.0, f"baseline expected 0, got {info['baseline']}"
    assert info['signal'] == 1.0, f"signal clipped to 1, got {info['signal']}"
    assert info['updated_count'] == 2
    # rec1: 0.9*0.8 + 0.1*1 = 0.82
    # rec2: 0.9*0.5 + 0.1*1 = 0.55
    assert abs(rec1.confidence - 0.82) < 1e-9, f"rec1 conf={rec1.confidence}"
    assert abs(rec2.confidence - 0.55) < 1e-9, f"rec2 conf={rec2.confidence}"
    assert adapter._episode_retrievals['safenutri'] == []
    print(f"\nEMA upward: baseline={info['baseline']}, signal={info['signal']:.3f}")
    print(f"  rec1 {before1:.3f} -> {rec1.confidence:.3f} (expected 0.820)")
    print(f"  rec2 {before2:.3f} -> {rec2.confidence:.3f} (expected 0.550)")

    # ---- Follow-up: lower reward, confidence should drop ----
    adapter.track_retrieval('safenutri', [rec1])
    before1 = rec1.confidence
    # Baseline buffer now has [2.0], so baseline=2.0 for this call.
    # Reward=0.5 → signal = (0.5 - 2.0) / 2.0 = -0.75
    # rec1: 0.9*0.82 + 0.1*(-0.75) = 0.738 - 0.075 = 0.663
    info2 = adapter.update_confidence('safenutri', episode_reward=0.5)
    assert info2['baseline'] == 2.0
    assert info2['signal'] < 0, f"signal should be negative, got {info2['signal']}"
    assert rec1.confidence < before1, \
        f"confidence should drop; before={before1}, after={rec1.confidence}"
    print(f"\nEMA downward: baseline={info2['baseline']}, signal={info2['signal']:+.3f}")
    print(f"  rec1 {before1:.3f} -> {rec1.confidence:.3f} (dropped — OK)")

    # ---- Deduplication: same rec twice ----
    adapter.track_retrieval('safenutri', [rec2, rec2])
    info3 = adapter.update_confidence('safenutri', episode_reward=1.0)
    assert info3['updated_count'] == 1, \
        f"dedup failed: expected 1 update, got {info3['updated_count']}"
    print(f"\nDedup: 2 identical retrievals -> updated_count={info3['updated_count']} — OK")

    # ---- to_transfer_result ----
    tr = adapter.to_transfer_result([rec1, rec2], [0.8, 0.4], gate_active=False)
    assert len(tr.bias_vector) == 8
    assert len(tr.retrieved_memories) == 2
    assert tr.retrieval_scores == [0.8, 0.4]
    assert tr.gate_active is False
    print(f"\nTransferResult: bias_len={len(tr.bias_vector)}, "
          f"records={len(tr.retrieved_memories)}, gate={tr.gate_active} — OK")

    print("\nTransferAdapter OK: bias vector, EMA update, and dedup all work")
