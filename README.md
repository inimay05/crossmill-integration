# CrossMill — Integration Layer

**Cross-Industry RL Platform · Unified Entry Point · Active Memory Layer**

CrossMill is a cross-industry reinforcement learning platform that connects
two POMDP industrial environments — SafeNutri (orange juice pasteurization)
and MegaForge (blast furnace steel production) — through a shared
CrossIndustryMemory layer that enables knowledge transfer between them.

The integration layer is the single entry point for training, evaluation,
and demo. It wires both environments and the memory layer together
without modifying either environment's internals.

Part of the CrossMill family, built for the OpenEnv AI Hackathon 2026
(Meta x Scaler School of Technology).

## Key Design Decisions

- Memory is active by default. CrossMillPlatform() with no arguments
  instantiates a fully configured CrossIndustryMemory with mode='cross' and
  bidirectional transfer on. Passing memory=None uses NoOpMemory (zero bias)
  for experimental baselines.
- Environments are untouched. The integration layer interacts with SafeNutri
  and MegaForge exclusively through their public OpenEnv interfaces.
- Observation augmentation is always on. The policy network always sees 23-dim
  (SafeNutri) or 26-dim (MegaForge). When memory is disabled, the bias vector
  is zeros. The observation dimension is identical across all memory modes.
- Config is the single source of truth. All platform-level constants live in
  crossmill/config.py.

## Quick Start

    cd crossmill-integration
    source venv/bin/activate
    python -m tests.test_platform

## License

MIT
