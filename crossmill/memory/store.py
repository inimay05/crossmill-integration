from collections import deque
from typing import Optional
from crossmill.models import MemoryRecord, MemoryConfig
from crossmill.memory.config import (
    EPISODIC_BUFFER_SIZE,
    POSITIVE_REWARD_THRESHOLD,
    NEGATIVE_REWARD_THRESHOLD,
    NEGATIVE_PROMOTION_THRESHOLD,
    PROMOTION_CONFIDENCE_BONUS,
    CONFIDENCE_CEILING,
    DIFFICULTY_RANK,
    SAFETY_REGIME_BY_DIFFICULTY,
    ENVIRONMENTS,
)


class MemoryStore:
    """
    Two-tier memory storage.

    episodic[env_name]: deque of MemoryRecord, maxlen=EPISODIC_BUFFER_SIZE
    semantic[env_name]: list of MemoryRecord (unbounded, slow growth)
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        env_names = list(ENVIRONMENTS.keys())
        self.episodic: dict[str, deque] = {
            name: deque(maxlen=EPISODIC_BUFFER_SIZE) for name in env_names
        }
        self.semantic: dict[str, list[MemoryRecord]] = {
            name: [] for name in env_names
        }
        self._timestamp_counter = 0   # global step at time of storage

    def store(self, record: MemoryRecord) -> bool:
        """
        Add a record to the episodic buffer if its reward signal is meaningful.

        Returns True if the record was stored, False if it was filtered out.

        Filter rule:
          - If record.outcome_reward >= POSITIVE_REWARD_THRESHOLD (0.01):
              store as a positive memory (is_negative=False)
          - If record.outcome_reward <= NEGATIVE_REWARD_THRESHOLD (-0.5):
              mark is_negative=True and store
          - Otherwise: skip (near-zero-reward steps carry no transfer info)
        """
        if record.outcome_reward >= POSITIVE_REWARD_THRESHOLD:
            record.is_negative = False
        elif record.outcome_reward <= NEGATIVE_REWARD_THRESHOLD:
            record.is_negative = True
        else:
            return False
        self._timestamp_counter += 1
        record.timestamp = self._timestamp_counter
        self.episodic[record.env_id].append(record)
        return True

    def _required_threshold(self, difficulty_group: list[str]) -> int:
        """
        Return the promotion threshold for a group of episodic entries with
        possibly mixed difficulties. Uses the hardest difficulty's threshold.
        """
        thresholds = self.config.promotion_thresholds
        hardest = max(difficulty_group, key=lambda d: DIFFICULTY_RANK[d])
        return thresholds[hardest]

    def _group_candidates(self, env_name: str, is_negative: bool) -> dict:
        """
        Group episodic entries by action_pattern for possible promotion.

        Returns {pattern_label: [MemoryRecord, ...]} of matching-sign entries.
        """
        buffer = self.episodic[env_name]
        groups: dict[str, list[MemoryRecord]] = {}
        for rec in buffer:
            if rec.is_negative != is_negative:
                continue
            if is_negative:
                if rec.outcome_reward > NEGATIVE_PROMOTION_THRESHOLD:
                    continue
            else:
                if rec.outcome_reward <= 0:
                    continue
            groups.setdefault(rec.action_pattern, []).append(rec)
        return groups

    def _promote_group(self, env_name: str, records: list[MemoryRecord],
                       is_negative: bool) -> MemoryRecord:
        """
        Merge a group of episodic entries into one semantic MemoryRecord.

        - abstract_state: element-wise mean
        - outcome_reward: mean
        - confidence: mean + PROMOTION_CONFIDENCE_BONUS, capped at CONFIDENCE_CEILING
        - provenance: union of contributing episode IDs
        - access_count: 0 (fresh for retrieval)
        - task_difficulty: hardest difficulty present
        """
        n = len(records)
        avg_abstract = [sum(r.abstract_state[i] for r in records) / n
                        for i in range(len(records[0].abstract_state))]
        avg_reward = sum(r.outcome_reward for r in records) / n
        avg_conf = sum(r.confidence for r in records) / n
        new_conf = min(CONFIDENCE_CEILING, avg_conf + PROMOTION_CONFIDENCE_BONUS)
        prov: list[str] = []
        for r in records:
            prov.extend(r.provenance)
        difficulties = [r.task_difficulty for r in records]
        hardest = max(difficulties, key=lambda d: DIFFICULTY_RANK[d])
        return MemoryRecord(
            env_id=env_name,
            abstract_state=avg_abstract,
            action_pattern=records[0].action_pattern,
            outcome_reward=avg_reward,
            is_negative=is_negative,
            safety_regime=SAFETY_REGIME_BY_DIFFICULTY[hardest],
            confidence=new_conf,
            access_count=0,
            task_difficulty=hardest,
            provenance=prov,
            timestamp=self._timestamp_counter,
        )

    def try_promote(self, env_name: str) -> list[MemoryRecord]:
        """
        Check all promotion candidates for env_name (both positive and negative).
        Fire promotions where the difficulty-aware threshold is met.

        Returns the list of newly promoted semantic records (possibly empty).
        """
        newly_promoted: list[MemoryRecord] = []
        for is_neg in (False, True):
            groups = self._group_candidates(env_name, is_negative=is_neg)
            for pattern_label, recs in groups.items():
                diffs = [r.task_difficulty for r in recs]
                threshold = self._required_threshold(diffs)
                if len(recs) >= threshold:
                    new_record = self._promote_group(env_name, recs, is_neg)
                    self.semantic[env_name].append(new_record)
                    newly_promoted.append(new_record)
                    for r in recs:
                        try:
                            self.episodic[env_name].remove(r)
                        except ValueError:
                            pass
        return newly_promoted

    def semantic_entries(self, env_name: str,
                         min_confidence: float = 0.0) -> list[MemoryRecord]:
        """
        Return semantic entries for env_name, filtered by confidence floor.
        Does NOT modify the entries (read-only view).
        """
        return [r for r in self.semantic[env_name]
                if r.confidence >= min_confidence]

    def stats(self) -> dict:
        """Return diagnostic counts."""
        env_names = list(self.episodic.keys())
        return {
            'episodic_count': {n: len(self.episodic[n]) for n in env_names},
            'semantic_count': {n: len(self.semantic[n]) for n in env_names},
            'confidence_distribution': [
                r.confidence for n in env_names for r in self.semantic[n]
            ],
        }


if __name__ == '__main__':
    def make_rec(env='safenutri', pattern='gradual_ramp', reward=0.1,
                 difficulty='easy', prov_id='e0'):
        return MemoryRecord(
            env_id=env,
            abstract_state=[0.5] * 8,
            action_pattern=pattern,
            outcome_reward=reward,
            safety_regime=SAFETY_REGIME_BY_DIFFICULTY[difficulty],
            confidence=0.5,
            task_difficulty=difficulty,
            provenance=[prov_id],
        )

    cfg = MemoryConfig()
    store = MemoryStore(cfg)

    # --- 5 positive easy entries (gradual_ramp) ---
    for i in range(5):
        ok = store.store(make_rec(prov_id=f'ep_e_{i}'))
        assert ok, "expected store() to accept positive reward"
    assert store.stats()['episodic_count']['safenutri'] == 5, \
        f"expected 5 episodic, got {store.stats()['episodic_count']['safenutri']}"
    print(f"After 5 stores: episodic_count = {store.stats()['episodic_count']['safenutri']}")

    promoted = store.try_promote('safenutri')
    assert len(promoted) == 1, f"expected 1 promotion, got {len(promoted)}"
    assert store.stats()['semantic_count']['safenutri'] == 1
    assert store.stats()['episodic_count']['safenutri'] == 0
    print(f"Promotion fired: semantic={store.stats()['semantic_count']['safenutri']}, "
          f"episodic={store.stats()['episodic_count']['safenutri']}")

    # --- 2 entries (below easy threshold of 3) ---
    for i in range(2):
        store.store(make_rec(pattern='hold_steady', prov_id=f'ep_h_{i}'))
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 0, "2 entries should NOT promote (threshold=3)"
    assert store.stats()['episodic_count']['safenutri'] == 2
    print(f"2 entries: no promotion (correct). episodic={store.stats()['episodic_count']['safenutri']}")

    # --- 3 hard entries (hard threshold is 5) — no promotion ---
    for i in range(3):
        store.store(make_rec(pattern='cautious_explore', difficulty='hard',
                             prov_id=f'ep_hard_{i}'))
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 0, "3 hard entries should NOT promote (threshold=5)"
    print(f"3 hard entries: no promotion (correct).")

    # --- 2 more hard entries (5 total) — should promote ---
    for i in range(2):
        store.store(make_rec(pattern='cautious_explore', difficulty='hard',
                             prov_id=f'ep_hard_{i+3}'))
    prev_sem = store.stats()['semantic_count']['safenutri']
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 1, f"expected 1 hard promotion, got {len(promoted)}"
    assert store.stats()['semantic_count']['safenutri'] == prev_sem + 1
    print(f"5 hard entries: 1 promotion fired. semantic={store.stats()['semantic_count']['safenutri']}")

    # --- Mixed difficulty: 2 easy + 2 hard of a new pattern ---
    for i in range(2):
        store.store(make_rec(pattern='emergency_response', difficulty='easy',
                             reward=0.2, prov_id=f'mix_e_{i}'))
    for i in range(2):
        store.store(make_rec(pattern='emergency_response', difficulty='hard',
                             reward=0.2, prov_id=f'mix_h_{i}'))
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 0, \
        "mixed 2 easy + 2 hard should NOT promote (threshold=hardest=5, only 4)"
    print(f"Mixed 2E+2H: no promotion (correct — threshold uses hardest=5).")

    # --- Negative memories: 3 rapid_correction easy entries with reward=-1.0 ---
    for i in range(3):
        ok = store.store(make_rec(pattern='rapid_correction',
                                   difficulty='easy', reward=-1.0,
                                   prov_id=f'neg_{i}'))
        assert ok
    prev_sem = store.stats()['semantic_count']['safenutri']
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 1, f"expected 1 negative promotion, got {len(promoted)}"
    assert promoted[0].is_negative is True, "negative promotion must set is_negative"
    assert store.stats()['semantic_count']['safenutri'] == prev_sem + 1
    print(f"3 negative entries: 1 negative promotion (is_negative={promoted[0].is_negative}).")

    # --- Below-threshold filter: reward=0.001 should be filtered out ---
    prev_ep = store.stats()['episodic_count']['safenutri']
    ok = store.store(make_rec(pattern='gradual_ramp', reward=0.001,
                              prov_id='filtered'))
    assert ok is False, "store() should reject reward=0.001 (below threshold)"
    assert store.stats()['episodic_count']['safenutri'] == prev_ep, \
        "episodic count must not change when store() returns False"
    print(f"Filter test: reward=0.001 rejected (store returned {ok}).")

    print("\nMemoryStore OK: promotion, thresholds, and filtering all work")
