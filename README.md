# CrossMill — Cross-Industry Reinforcement Learning Platform

> The integration layer wiring SafeNutri and MegaForge with optional CrossIndustryMemory transfer.


---

## What CrossMill Is

CrossMill is a research platform that connects two industrial POMDP reinforcement learning environments — SafeNutri (orange juice HTST pasteurization) and MegaForge (blast furnace steel production) — through a shared cross-industry memory layer. The central claim: a control lesson learned on a juice pasteurization line can meaningfully accelerate learning in a blast furnace, and vice versa.

At first glance, the two environments look completely unrelated. Under the hood they share the same structural control challenge: navigate a thermally inertial, partially observable system toward a multi-objective optimum without crossing hard safety boundaries. That shared geometry is what makes cross-industry transfer non-trivial and operationally meaningful, rather than a domain trick.

The platform is built around a precise, falsifiable headline metric: `transfer_gain = cross_score − local_score`. A positive value means cross-industry memory added something that single-environment replay alone could not. The grader validates this claim against four anti-reward-hacking gates before it counts as a clean result.

---

## Architecture

CrossMill uses a two-tier agent design:

- **LLM Strategist** (`Qwen2.5-3B`, GRPO via TRL + LoRA): reads factory state and emits a 4D `strategy_bias` — a high-level directional nudge (push harder, protect equipment, conserve energy, etc.). Re-reads state every N steps as factory conditions evolve. Does not control valves or setpoints directly.
- **RecurrentPPO Controller** (`MlpLstmPolicy`, sb3-contrib): receives raw observation + 8D memory bias + 4D strategy bias and produces precise numeric actions. Maintains a short-term memory of recent steps to understand system trajectory, not just current state.

Final policy input sizes: SafeNutri `15 + 8 + 4 = 27`, MegaForge `18 + 8 + 4 = 30`.

```text
CrossMillPlatform
├── SafeNutriEnv  (importlib loaded, never modified)
├── MegaForgeEnv  (importlib loaded, never modified)
├── CrossIndustryMemory
│   ├── MemoryStore      (episodic ring buffer + semantic store)
│   ├── Retriever        (cosine similarity + confidence warmup gate)
│   └── TransferAdapter  (bias vector + EMA confidence updates)
└── CrossMillGymShim → RecurrentPPO (MlpLstmPolicy)
                     ← strategy_bias ← Qwen2.5-3B (GRPO + LoRA)
```

---

## The CrossIndustry Memory Layer

The memory layer is the architectural heart of CrossMill. It is what separates the platform from a collection of independent simulations.

### Two-tier store

- **Episodic buffer**: ring buffer, 500 entries per environment. Stores raw step transitions where reward is informative: positive memory if reward `≥ 0.01`, negative memory if reward `≤ −0.5`.
- **Semantic store**: promoted patterns, confidence-scored. Promotion thresholds are difficulty-aware — Easy: 3 confirmations, Medium: 4, Hard: 5. Harder tasks demand stronger evidence before a pattern is trusted.

Both successful and failure patterns are stored. Negative memories promote to the semantic store and are sign-inverted during bias construction, so the agent learns what to avoid, not only what to repeat.

### 8D abstract state projection

Every observation from both environments is projected into the same 8-dimensional abstract basis before storage and retrieval:

| Dimension | Meaning |
|-----------|---------|
| `process_temperature` | Normalised thermal level |
| `temperature_rate` | Rate of thermal change |
| `quality_risk` | Risk of quality degradation |
| `safety_margin` | Distance from hard constraint violation |
| `energy_intensity` | Current energy consumption |
| `equipment_condition` | Equipment health state |
| `process_progress` | Progress through episode |
| `throughput_pressure` | Production rate pressure |

This shared abstract basis makes cosine similarity meaningful across industries with completely different physical variables.

### Retrieval and bias injection

Retrieval score: `0.7 × cosine_similarity + 0.3 × confidence`. Top-3 matches are returned and combined into an 8D bias vector. The bias is hard-clipped to `±0.5` before concatenation — memory nudges the agent, never overrides the physics of the live environment.

Confidence warmup gate: starts at `0.4`, decays to `0.1` over the first 50,000 global steps. This prevents unstable early transfer before memories are trustworthy.

Confidence updates via EMA after every episode: `new_conf = (1 − α) × old + α × signal`, α = 0.1, bounded to `[0.05, 1.0]`. Signal is reward relative to a rolling 10-episode baseline.

### Three memory modes

| Mode | Description |
|------|-------------|
| `none` | No memory. Bias vector zeroed. Baseline equivalent to all existing industrial RL approaches. |
| `local` | Memory populated only from the same environment's own episodes. |
| `cross` | Memory shared across both SafeNutri and MegaForge episodes. |

`transfer_gain = cross_score − local_score` is the headline. Positive value: cross-industry memory added real value. The grader validates this with four anti-hacking gates before it counts.

---

## The Unified Grader and Anti-Reward-Hacking Layer

The grader never re-runs training — it reads logged results only, then applies validation flags before reporting final scores.

### Anti-hacking flags

| Flag | Condition | Effect |
|------|-----------|--------|
| `SAFETY_WARNING` | `safety_violation_rate > 0` | Warning |
| `CATASTROPHIC_FAIL` | `catastrophic_rate > 0` | Adjusted score forced to **0** |
| `QUALITY_LOW` | SafeNutri: low vitamin C retention · MegaForge: high carbon error | Warning |
| `HIGH_VARIANCE` | Coefficient of variation (`std/mean`) > 0.8 | Warning |
| `COMPOUND_FAIL` | `SAFETY_WARNING` + `QUALITY_LOW` both active | Adjusted score **halved** |

### Delta decomposition

All deltas are computed on raw `grader_score`, not adjusted scores:

- `local_gain = local_score − none_score`
- `cross_gain = cross_score − none_score`
- `transfer_gain = cross_score − local_score`

`transfer_gain` separates replay benefit from genuine cross-industry transfer and makes the claim falsifiable.

---

## Environments

| | SafeNutri | MegaForge |
|---|---|---|
| **Domain** | Orange juice HTST pasteurization | Blast furnace steel production |
| **State** | 15-dim normalised | 18-dim normalised |
| **Actions** | 8-dim (6 continuous + 2 discrete) | 10-dim (8 continuous + 2 discrete) |
| **Episode lengths** | Easy 300 / Medium 800 / Hard 2000 | Easy 200 / Medium 600 / Hard 2000 |
| **Grader targets** | ≥ 0.85 / ≥ 0.78 / ≥ 0.72 | ≥ 0.88 / ≥ 0.80 / ≥ 0.74 |
| **POMDP features** | Lab delay, partial obs, season switches | Assay delay, tapping gates, maintenance window |
| **GitHub** | [crossmill-safenutri](https://github.com/inimay05/crossmill-safenutri) | [crossmill-megaforge](https://github.com/inimay05/crossmill-megaforge) |
| **HF Space** | [kolaai/crossmill-safenutri](https://huggingface.co/spaces/kolaai/crossmill-safenutri) | [kolaai/crossmill-megaforge](https://huggingface.co/spaces/kolaai/crossmill-megaforge) |

Both tasks are true POMDPs with delayed measurements, hidden slow state, and stochastic regime shifts — which is why CrossMill trains with RecurrentPPO (MlpLstmPolicy) rather than feedforward PPO.

---

## Repository Structure

```
crossmill-integration/
├── crossmill/               # Platform core
│   ├── platform.py          # CrossMillPlatform: loads both envs, wires memory layer
│   ├── augmentation.py      # Obs augmentation: raw + memory bias + strategy bias
│   ├── gym_shim.py          # Gymnasium-compatible wrapper for SB3
│   ├── grader_validation.py # Anti-hacking grader: flags + delta decomposition
│   └── memory/
│       ├── store.py         # MemoryStore: episodic ring buffer + semantic store
│       ├── retriever.py     # Retriever: cosine similarity + confidence warmup
│       └── transfer.py      # TransferAdapter: bias vector + EMA updates
├── scripts/
│   ├── train.py             # RecurrentPPO controller training
│   └── train_llm_grpo.py    # LLM strategist training (GRPO + LoRA)
├── tests/
│   ├── test_platform.py     # 7 integration tests
│   ├── test_memory.py       # 24 memory tests
│   └── test_grader.py       # 13 grader tests
├── notebooks/               # Experiment notebooks
└── docs/                    # Design documents and internal reports
```

---

## Quick Start

```bash
git clone https://github.com/inimay05/crossmill-safenutri
git clone https://github.com/inimay05/crossmill-megaforge
git clone https://github.com/inimay05/crossmill-integration

cd crossmill-integration
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH="$PYTHONPATH:$(pwd)/../crossmill-safenutri:$(pwd)/../crossmill-megaforge"

# Run full test suite
python -m tests.test_platform   # 7/7
python -m tests.test_memory     # 24/24
python -m tests.test_grader     # 13/13
```

Train LLM strategist first, then controller:

```bash
python scripts/train_llm_grpo.py

python scripts/train.py --env safenutri --task easy --memory_mode cross --seed 42 --llm_strategist
```

Then grade results:

```bash
python -m crossmill.grader --env safenutri --task easy
```

---

## Live Demo

The CrossMill Gradio Space is live at:
**https://huggingface.co/spaces/kolaai/crossmill-integration**

It includes the CrossMill story page and a Results Dashboard that populates once training artifacts are present.

---

## Test Suite

- **7 integration tests** (`tests/test_platform.py`): observation dimensions, reproducibility, memory isolation, episode lifecycle, augmentation clipping and field ordering
- **24 memory tests** (`tests/test_memory.py`): abstraction, action classification, promotion thresholds, retrieval modes and direction filters, confidence warmup, EMA confidence updates, interface compliance
- **13 grader tests** (`tests/test_grader.py`): all anti-hacking flag paths, compound gate behaviour, delta computation on raw scores, summary schema validation
