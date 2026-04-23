import math
from crossmill.models import MemoryRecord, MemoryConfig
from crossmill.memory.store import MemoryStore
from crossmill.memory.config import (
    SIMILARITY_WEIGHT,
    CONFIDENCE_WEIGHT,
    ABSTRACT_DIM,
)


class Retriever:
    """
    Queries the MemoryStore and returns the top-k most relevant memories
    given a current abstract state.
    """

    def __init__(self, store: MemoryStore, config: MemoryConfig):
        self.store = store
        self.config = config

    def _current_min_confidence(self, global_step: int) -> float:
        """
        Linearly decay the confidence floor from initial_min_confidence to
        min_confidence over confidence_warmup_steps. If initial == min,
        warmup is disabled (constant floor).
        """
        init = self.config.initial_min_confidence
        final = self.config.min_confidence
        warmup = self.config.confidence_warmup_steps
        if warmup <= 0 or init == final:
            return final
        decay_progress = min(1.0, global_step / warmup)
        return init - (init - final) * decay_progress

    def _eligible_env_ids(self, current_env: str) -> list[str]:
        """
        Apply mode and transfer_direction filters.
        Returns the list of env_ids whose semantic store should be queried.
        """
        mode = self.config.mode
        direction = self.config.transfer_direction
        if mode == 'none':
            return []
        if mode == 'local':
            return [current_env]
        # mode == 'cross'
        if direction == 'bidirectional':
            return [env for env in self.store.semantic.keys()
                    if env != current_env]
        if direction == 'steel_to_food':
            return ['megaforge'] if current_env == 'safenutri' else []
        if direction == 'food_to_steel':
            return ['safenutri'] if current_env == 'megaforge' else []
        return []

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Standard cosine similarity on two equal-length vectors.
        Returns 0.0 if either vector has zero magnitude.
        Result is clipped to [0, 1] because all inputs are in [0, 1] — we treat
        anti-correlation the same as uncorrelation for retrieval purposes.
        """
        if len(a) != len(b):
            raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x*x for x in a))
        mag_b = math.sqrt(sum(y*y for y in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        sim = dot / (mag_a * mag_b)
        return max(0.0, min(1.0, sim))

    def retrieve(self, current_env: str, abstract_state: list[float],
                 global_step: int = 0) -> tuple[list[MemoryRecord], list[float], bool]:
        """
        Return (top_k_records, top_k_scores, gate_active).

        gate_active is True when the confidence floor filtered out ALL memories
        (either the semantic store is empty or nothing passed the floor).

        Args:
          current_env: 'safenutri' or 'megaforge'
          abstract_state: 8-dim list[float], all in [0, 1]
          global_step: used to compute the current confidence floor

        Behaviour:
          1. Determine eligible env_ids from mode + transfer_direction
          2. Collect semantic records from those env_ids whose confidence
             >= current_min_confidence(global_step)
          3. Score each = 0.7 * cosine_similarity + 0.3 * confidence
          4. Return top-k by score, where k = config.top_k
          5. If mode='none' or no eligible envs or no records pass, return
             ([], [], True)  — gate_active indicates no suggestions given.
        """
        if len(abstract_state) != ABSTRACT_DIM:
            raise ValueError(f"abstract_state length {len(abstract_state)}, "
                             f"expected {ABSTRACT_DIM}")

        eligible_envs = self._eligible_env_ids(current_env)
        if not eligible_envs:
            return ([], [], True)

        floor = self._current_min_confidence(global_step)

        candidates: list[MemoryRecord] = []
        for env_id in eligible_envs:
            candidates.extend(
                self.store.semantic_entries(env_id, min_confidence=floor)
            )

        if not candidates:
            return ([], [], True)

        scored: list[tuple[MemoryRecord, float]] = []
        for rec in candidates:
            sim = self.cosine_similarity(abstract_state, rec.abstract_state)
            score = SIMILARITY_WEIGHT * sim + CONFIDENCE_WEIGHT * rec.confidence
            scored.append((rec, score))

        scored.sort(key=lambda t: t[1], reverse=True)
        top = scored[: self.config.top_k]

        for rec, _ in top:
            rec.access_count += 1

        records = [t[0] for t in top]
        scores = [t[1] for t in top]
        gate_active = len(records) == 0
        return (records, scores, gate_active)


if __name__ == '__main__':
    from crossmill.memory.config import SAFETY_REGIME_BY_DIFFICULTY

    def make_rec(env, pattern, state, conf, difficulty='easy', reward=0.2):
        return MemoryRecord(
            env_id=env,
            abstract_state=state,
            action_pattern=pattern,
            outcome_reward=reward,
            safety_regime=SAFETY_REGIME_BY_DIFFICULTY[difficulty],
            confidence=conf,
            task_difficulty=difficulty,
            provenance=['manual'],
        )

    # ---- Setup: populate semantic tier directly ----
    cfg = MemoryConfig()
    store = MemoryStore(cfg)

    mf_gradual_state = [0.8, 0.1, 0.2, 0.3, 0.5, 0.7, 0.4, 0.3]
    mf_rapid_state   = [0.3, 0.9, 0.7, 0.1, 0.6, 0.5, 0.5, 0.4]
    sn_gradual_state = [0.7, 0.2, 0.25, 0.35, 0.45, 0.75, 0.4, 0.35]

    mf_gradual = make_rec('megaforge', 'gradual_ramp', mf_gradual_state, 0.9)
    mf_rapid   = make_rec('megaforge', 'rapid_correction', mf_rapid_state, 0.6)
    sn_gradual = make_rec('safenutri', 'gradual_ramp', sn_gradual_state, 0.8)

    store.semantic['megaforge'].extend([mf_gradual, mf_rapid])
    store.semantic['safenutri'].append(sn_gradual)

    # ---- Test 1: cross / bidirectional from safenutri, close to mf_gradual ----
    r = Retriever(store, cfg)
    query = [0.79, 0.12, 0.21, 0.32, 0.48, 0.72, 0.41, 0.31]  # ~mf_gradual
    records, scores, gate = r.retrieve('safenutri', query, global_step=100000)
    print(f"Test 1 (cross/bidir from safenutri, ~mf_gradual):")
    print(f"  top-{len(records)} with scores={[f'{s:.3f}' for s in scores]}")
    assert len(records) >= 1, "expected at least 1 record"
    assert records[0].action_pattern == 'gradual_ramp', \
        f"expected gradual_ramp, got {records[0].action_pattern}"
    assert records[0].env_id == 'megaforge', \
        f"expected megaforge, got {records[0].env_id}"
    assert not gate
    print(f"  top-1: {records[0].env_id}/{records[0].action_pattern}  score={scores[0]:.3f}  — OK")

    # ---- Test 2: mode='none' ----
    none_cfg = MemoryConfig(mode='none')
    r2 = Retriever(store, none_cfg)
    recs, scs, gate = r2.retrieve('safenutri', query, global_step=100000)
    assert recs == [] and gate is True
    print(f"Test 2 (mode='none'): empty + gate_active=True — OK")

    # ---- Test 3: mode='local' from safenutri ----
    local_cfg = MemoryConfig(mode='local')
    r3 = Retriever(store, local_cfg)
    recs, scs, gate = r3.retrieve('safenutri', query, global_step=100000)
    assert len(recs) >= 1
    for rec in recs:
        assert rec.env_id == 'safenutri', \
            f"local mode leaked {rec.env_id}"
    print(f"Test 3 (mode='local' from safenutri): only safenutri returned — OK")

    # ---- Test 4: cross/steel_to_food from megaforge -> empty ----
    s2f_cfg = MemoryConfig(mode='cross', transfer_direction='steel_to_food')
    r4 = Retriever(store, s2f_cfg)
    recs, scs, gate = r4.retrieve('megaforge', query, global_step=100000)
    assert recs == [] and gate is True, \
        f"steel_to_food from megaforge must be empty, got {len(recs)}"
    print(f"Test 4 (steel_to_food from megaforge): empty (wrong direction) — OK")

    # ---- Test 5: cross/steel_to_food from safenutri -> megaforge records ----
    recs, scs, gate = r4.retrieve('safenutri', query, global_step=100000)
    assert len(recs) >= 1
    for rec in recs:
        assert rec.env_id == 'megaforge', \
            f"steel_to_food into safenutri must return megaforge, got {rec.env_id}"
    print(f"Test 5 (steel_to_food from safenutri): {len(recs)} megaforge records — OK")

    # ---- Test 6: warmup at step=0, floor=0.9 ----
    warmup_cfg = MemoryConfig(
        mode='cross', transfer_direction='bidirectional',
        min_confidence=0.1, initial_min_confidence=0.9,
        confidence_warmup_steps=1000,
    )
    r6 = Retriever(store, warmup_cfg)
    recs0, _, _ = r6.retrieve('safenutri', query, global_step=0)
    # Only mf_gradual (conf=0.9) passes floor=0.9; mf_rapid (0.6) does not
    assert len(recs0) == 1 and recs0[0].confidence == 0.9, \
        f"at step=0, only conf=0.9 should pass; got {[r.confidence for r in recs0]}"
    print(f"Test 6 (warmup step=0, floor=0.9): 1 record (conf=0.9) — OK")

    # ---- Test 7: warmup at step=1000, floor=0.1 ----
    recs1k, _, _ = r6.retrieve('safenutri', query, global_step=1000)
    assert len(recs1k) == 2, \
        f"at step=1000, both megaforge records should pass; got {len(recs1k)}"
    print(f"Test 7 (warmup step=1000, floor=0.1): {len(recs1k)} records — OK")

    # ---- Test 8: gate_active when floor filters all ----
    gate_cfg = MemoryConfig(
        mode='cross', transfer_direction='bidirectional',
        min_confidence=0.95, initial_min_confidence=0.95,
        confidence_warmup_steps=0,
    )
    r8 = Retriever(store, gate_cfg)
    recs, scs, gate = r8.retrieve('safenutri', query, global_step=100000)
    assert recs == [] and gate is True, \
        f"floor=0.95 should filter all; got {len(recs)}"
    print(f"Test 8 (floor=0.95): empty + gate_active=True — OK")

    # ---- Test 9: access_count increments on retrieval ----
    before = mf_gradual.access_count
    recs, _, _ = r.retrieve('safenutri', query, global_step=100000)
    after = mf_gradual.access_count
    assert after == before + 1, \
        f"access_count: expected {before+1}, got {after}"
    print(f"Test 9 (access_count): {before} -> {after} — OK")

    print("\nRetriever OK: mode filtering, direction, warmup, and top-k all work")
