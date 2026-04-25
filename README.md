---
title: CrossMill
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: true
---

# CrossMill — Cross-Industry RL Platform

**Teaching Steel to Learn from Juice**

CrossMill is a cross-industry reinforcement learning platform that connects
two production-fidelity POMDP industrial environments —
**SafeNutri** (orange juice pasteurisation) and **MegaForge** (blast furnace steelmaking) —
through a shared `CrossIndustryMemory` layer that enables genuine knowledge
transfer between unrelated manufacturing sectors.

The central insight is structural isomorphism: Vitamin C degradation and
refractory wear obey the same physics. Temperature ramp control in a
pasteuriser and blast temperature control in a furnace are the same
optimisation problem at different scales. A policy that learns one already
knows something about the other.

Built for the **OpenEnv AI Hackathon 2026** · Meta × Scaler School of Technology.

---

## What's in this Space

| Tab | Contents |
|---|---|
| 🏭 CrossMill Story | Interactive story-driven landing page explaining the platform |
| 📊 Results Dashboard | Live training results table + reward curves (populated after training) |

---

## Key Technical Contributions

- **Two physics-accurate Gymnasium environments** — SafeNutri (15-dim state, 8 actions,
  Arrhenius kinetics, FDA constraints) and MegaForge (18-dim state, 10 actions,
  blast thermochemistry, ±0.05% carbon target)
- **CrossIndustryMemory** — episodic → semantic promotion with confidence-gated
  cross-domain retrieval (`0.7 × cosine_similarity + 0.3 × confidence`)
- **Three memory modes** — `none` / `local` / `cross` for clean ablation
- **Hierarchical dual-agent** — Qwen2.5-3B LLM as plant manager feeds
  `strategy_bias` into RecurrentPPO + LSTM controller
- **Anti-reward-hacking flags** — `SAFETY_WARNING`, `CATASTROPHIC_FAIL`,
  `QUALITY_LOW`, `HIGH_VARIANCE`, `COMPOUND_FAIL`
- **Transfer gain metric** — `(R_cross − R_none) / |R_none|` as primary
  cross-industry evaluation signal

---

## Environments

### 🍊 SafeNutri — OJ Pasteurisation

| Property | Value |
|---|---|
| State dimensions | 15 |
| Action dimensions | 8 |
| Thermal model | Arrhenius kinetics |
| Safety constraint | FDA §21 CFR pathogen kill |
| Optimisation target | Vitamin C retention |
| POMDP | Partial obs, delayed sensors, noise |
| Difficulty | Easy / Medium / Hard |

### 🏭 MegaForge — Blast Furnace Steelmaking

| Property | Value |
|---|---|
| State dimensions | 18 |
| Action dimensions | 10 |
| Carbon target | 4.2% ± 0.05% |
| Equipment value | $500M |
| Safety constraint | Thermal stress / refractory wear |
| POMDP | Partial obs, delayed sensors |
| Difficulty | Easy / Medium / Hard |

---

## Structural Isomorphism

| SafeNutri | MegaForge |
|---|---|
| Vitamin C degradation | Refractory wear |
| Temperature ramp control | Blast temperature control |
| FDA pathogen-kill limit | Thermal stress limit |
| Flow rate & pressure | Production throughput |
| Arrhenius kinetics | Blast thermochemistry |

---

## License

Apache 2.0
