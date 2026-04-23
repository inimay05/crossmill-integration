from typing import Callable
from crossmill.memory.config import ABSTRACT_DIM

# ---- ABSTRACT STATE DIMENSION ORDER ----
# Both abstraction functions MUST return values in this exact order:
#   0: process_temperature    — normalised process temperature [0, 1]
#   1: temperature_rate       — absolute temperature rate of change [0, 1]
#   2: quality_risk           — distance from target quality [0, 1]
#   3: safety_margin          — minimum distance to nearest safety limit [0, 1]
#   4: energy_intensity       — normalised energy consumption [0, 1]
#   5: equipment_condition    — equipment health [0, 1]
#   6: process_progress       — fraction of episode elapsed [0, 1]
#   7: throughput_pressure    — throughput relative to max capacity [0, 1]


def abstract_safenutri(obs_dict: dict, task_id: str = 'easy') -> list[float]:
    """
    Project a SafeNutri observation into the 8-dim abstract state.

    Args:
      obs_dict: SafeNutri normalised observation dict. All numeric state
        fields in [0, 1]. Expected keys include:
          temperature, temp_gradient, vitamin_c, contamination_risk,
          energy, equip_efficiency, time_in_process, flow_rate
      task_id: 'easy' / 'medium' / 'hard' — used to normalise process_progress
        against the correct episode length.

    Returns:
      list[float] of length 8, all in [0, 1].
    """
    # Episode lengths by task (derived from SafeNutri task configs)
    EPISODE_MAX_STEPS = {'easy': 300, 'medium': 800, 'hard': 2000}

    # Dimension 0: process_temperature (already normalised)
    process_temperature = float(obs_dict['temperature'])

    # Dimension 1: temperature_rate (already normalised, absolute value)
    temperature_rate = abs(float(obs_dict['temp_gradient']))

    # Dimension 2: quality_risk — complement of vitamin C retention
    quality_risk = 1.0 - float(obs_dict['vitamin_c'])

    # Dimension 3: safety_margin — minimum of (distance to contamination limit)
    # Normalised obs is [0,1]; contamination_risk of 1.0 means worst case (100)
    # so safety_margin = 1 - contamination_risk captures headroom
    # Combine with temperature margin: (1 - temperature) captures headroom to
    # the 95°C hard limit for normalised temperature near 1.0 (= 90°C).
    contam = float(obs_dict['contamination_risk'])
    temp = float(obs_dict['temperature'])
    safety_margin = min(1.0 - contam, 1.0 - temp * 0.95)
    safety_margin = max(0.0, min(1.0, safety_margin))

    # Dimension 4: energy_intensity (already normalised)
    energy_intensity = float(obs_dict['energy'])

    # Dimension 5: equipment_condition (already normalised)
    equipment_condition = float(obs_dict['equip_efficiency'])

    # Dimension 6: process_progress — fraction of episode elapsed
    # time_in_process is normalised over a fixed max of 600s per the SafeNutri
    # config RANGES. We scale further by episode length.
    time_s = float(obs_dict['time_in_process']) * 600.0  # denormalised seconds
    max_s = EPISODE_MAX_STEPS.get(task_id, 300) * 5.0  # 5s per step from config
    process_progress = min(1.0, time_s / max_s) if max_s > 0 else 0.0

    # Dimension 7: throughput_pressure (already normalised)
    throughput_pressure = float(obs_dict['flow_rate'])

    vec = [process_temperature, temperature_rate, quality_risk, safety_margin,
           energy_intensity, equipment_condition, process_progress,
           throughput_pressure]
    # Final safety clip — all values must be in [0, 1]
    return [max(0.0, min(1.0, float(v))) for v in vec]


def abstract_megaforge(obs_dict: dict, task_id: str = 'easy') -> list[float]:
    """
    Project a MegaForge observation into the 8-dim abstract state.

    Same 8 dimensions in the same order as abstract_safenutri.

    Args:
      obs_dict: MegaForge normalised observation dict. All numeric state
        fields in [0, 1]. Expected keys include:
          hot_metal_temp, carbon, wall_temp, thermal_stress, energy,
          equip_health, step_idx, production_rate
      task_id: 'easy' / 'medium' / 'hard' — used for process_progress.

    Returns:
      list[float] of length 8, all in [0, 1].
    """
    EPISODE_MAX_STEPS = {'easy': 200, 'medium': 600, 'hard': 2000}

    # Dimension 0: process_temperature (already normalised)
    process_temperature = float(obs_dict['hot_metal_temp'])

    # Dimension 1: temperature_rate — MegaForge does not expose a gradient
    # field directly. We use the blast_temp as a proxy for how aggressive the
    # heat input is — high blast temp drives temp changes. Alternative: use
    # |oxygen_flow - 0.5| to reflect active control intensity.
    blast_intensity = abs(float(obs_dict.get('blast_temp', 0.5)) - 0.5) * 2.0
    temperature_rate = min(1.0, blast_intensity)

    # Dimension 2: quality_risk — deviation from carbon target
    # Carbon target is 4.2% against range [3.8, 4.5] (width 0.7), so at
    # normalised position ~0.57. Use absolute distance.
    carbon_norm = float(obs_dict['carbon'])
    carbon_target_norm = (4.2 - 3.8) / (4.5 - 3.8)  # ≈ 0.571
    quality_risk = min(1.0, abs(carbon_norm - carbon_target_norm) / 0.3)

    # Dimension 3: safety_margin — minimum of wall_temp and thermal_stress
    # margins. Both are already normalised so 1.0 means at the limit.
    wall_margin = 1.0 - float(obs_dict['wall_temp'])
    stress_margin = 1.0 - float(obs_dict['thermal_stress'])
    safety_margin = max(0.0, min(wall_margin, stress_margin))

    # Dimension 4: energy_intensity (already normalised)
    energy_intensity = float(obs_dict['energy'])

    # Dimension 5: equipment_condition (already normalised)
    equipment_condition = float(obs_dict['equip_health'])

    # Dimension 6: process_progress — step_idx over episode length
    step_idx = int(obs_dict.get('step_idx', 0))
    max_steps = EPISODE_MAX_STEPS.get(task_id, 200)
    process_progress = min(1.0, step_idx / max_steps) if max_steps > 0 else 0.0

    # Dimension 7: throughput_pressure (already normalised)
    throughput_pressure = float(obs_dict['production_rate'])

    vec = [process_temperature, temperature_rate, quality_risk, safety_margin,
           energy_intensity, equipment_condition, process_progress,
           throughput_pressure]
    return [max(0.0, min(1.0, float(v))) for v in vec]


# ---- DISPATCHER ----
ABSTRACTION_FUNCTIONS: dict[str, Callable[[dict, str], list[float]]] = {
    'safenutri': abstract_safenutri,
    'megaforge': abstract_megaforge,
}


def abstract_observation(env_name: str, obs_dict: dict,
                         task_id: str = 'easy') -> list[float]:
    """
    Dispatch to the correct abstraction function for the given environment.
    """
    if env_name not in ABSTRACTION_FUNCTIONS:
        raise ValueError(f"Unknown env_name: {env_name!r}. "
                         f"Expected one of {list(ABSTRACTION_FUNCTIONS.keys())}")
    fn = ABSTRACTION_FUNCTIONS[env_name]
    result = fn(obs_dict, task_id=task_id)
    if len(result) != ABSTRACT_DIM:
        raise ValueError(f"Abstraction function for {env_name} returned "
                         f"{len(result)} dims, expected {ABSTRACT_DIM}")
    return result


if __name__ == '__main__':
    DIM_LABELS = [
        'process_temperature', 'temperature_rate', 'quality_risk',
        'safety_margin', 'energy_intensity', 'equipment_condition',
        'process_progress', 'throughput_pressure',
    ]

    # ---- SafeNutri: all fields at 0.5 ----
    sn_fields = [
        'temperature', 'temp_gradient', 'e_coli', 'salmonella', 'listeria',
        'vitamin_c', 'folate', 'thiamine', 'pH', 'brix', 'flow_rate',
        'energy', 'equip_efficiency', 'time_in_process', 'contamination_risk',
    ]
    sn_obs = {k: 0.5 for k in sn_fields}
    sn_result = abstract_safenutri(sn_obs, task_id='easy')
    assert len(sn_result) == 8, f"Expected 8, got {len(sn_result)}"
    assert all(0.0 <= v <= 1.0 for v in sn_result), f"Out of [0,1]: {sn_result}"

    print("SafeNutri (all=0.5, easy):")
    for label, val in zip(DIM_LABELS, sn_result):
        print(f"  {label:<22s} {val:.4f}")

    # ---- MegaForge: all fields at 0.5 ----
    mf_fields = [
        'hot_metal_temp', 'hearth_temp', 'blast_temp', 'oxygen_flow',
        'carbon', 'silicon', 'sulfur', 'top_pressure', 'co_co2_ratio',
        'coke_rate', 'ore_coke_ratio', 'energy', 'production_rate',
        'wall_temp', 'thermal_stress', 'slag_basicity', 'emissions_co2',
        'equip_health',
    ]
    mf_obs = {k: 0.5 for k in mf_fields}
    mf_obs['step_idx'] = 0
    mf_result = abstract_megaforge(mf_obs, task_id='easy')
    assert len(mf_result) == 8, f"Expected 8, got {len(mf_result)}"
    assert all(0.0 <= v <= 1.0 for v in mf_result), f"Out of [0,1]: {mf_result}"

    print("\nMegaForge (all=0.5, easy, step_idx=0):")
    for label, val in zip(DIM_LABELS, mf_result):
        print(f"  {label:<22s} {val:.4f}")

    # ---- Dispatcher ----
    d_sn = abstract_observation('safenutri', sn_obs, task_id='easy')
    assert len(d_sn) == 8
    d_mf = abstract_observation('megaforge', mf_obs, task_id='easy')
    assert len(d_mf) == 8
    print("\nDispatcher: both return 8-element vectors — OK")

    # ---- Extreme case: saturation ----
    extreme_obs = {k: 0.5 for k in sn_fields}
    extreme_obs['temperature'] = 1.0
    extreme_obs['contamination_risk'] = 1.0
    extreme_result = abstract_safenutri(extreme_obs, task_id='easy')
    assert extreme_result[3] == 0.0, (
        f"safety_margin should be 0.0 at saturation, got {extreme_result[3]}"
    )
    print(f"Extreme saturation: safety_margin = {extreme_result[3]:.4f} (expected 0.0) — OK")

    print("\nAbstraction OK: safenutri and megaforge both produce 8-dim vectors")
