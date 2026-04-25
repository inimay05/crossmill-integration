from __future__ import annotations

import numpy as np

import crossmill.config as config

# ---- Field ordering for observation-to-vector conversion ----
# These must match the field order in each environment's Observation model.
# The policy network sees values in this exact order.

SAFENUTRI_OBS_FIELDS = [
    'temperature', 'temp_gradient', 'e_coli', 'salmonella', 'listeria',
    'vitamin_c', 'folate', 'thiamine', 'pH', 'brix',
    'flow_rate', 'energy', 'equip_efficiency', 'time_in_process',
    'contamination_risk',
]  # 15 fields

MEGAFORGE_OBS_FIELDS = [
    'hot_metal_temp', 'hearth_temp', 'blast_temp', 'oxygen_flow',
    'carbon', 'silicon', 'sulfur', 'top_pressure', 'co_co2_ratio',
    'coke_rate', 'ore_coke_ratio', 'energy', 'production_rate',
    'wall_temp', 'thermal_stress', 'slag_basicity', 'emissions_co2',
    'equip_health',
]  # 18 fields

OBS_FIELDS = {
    'safenutri': SAFENUTRI_OBS_FIELDS,
    'megaforge': MEGAFORGE_OBS_FIELDS,
}


def obs_to_vector(env_name: str, obs) -> np.ndarray:
    """
    Convert a Pydantic Observation model to a 1D numpy float32 array.
    Extracts the numeric state fields in the canonical order defined above.

    Args:
      env_name: 'safenutri' or 'megaforge'
      obs: the Observation pydantic model returned by env.reset() or
           env.step().observation

    Returns:
      np.ndarray of shape (state_dim,) — 15 for safenutri, 18 for megaforge.
      All values should be in [0, 1] (normalised by the environment).
    """
    fields = OBS_FIELDS[env_name]
    if hasattr(obs, 'model_dump'):
        d = obs.model_dump()
    elif isinstance(obs, dict):
        d = obs
    else:
        raise TypeError(f'Cannot convert observation of type {type(obs)}')
    return np.array([float(d[f]) for f in fields], dtype=np.float32)


def augment_observation(
    raw_vector: np.ndarray,
    bias_vector: list[float] | np.ndarray,
    env_name: str,
) -> np.ndarray:
    """
    Append the memory bias vector to the raw observation vector.

    Args:
      raw_vector: np.ndarray of shape (state_dim,) — from obs_to_vector
      bias_vector: list or array of length BIAS_VECTOR_DIM (8)
      env_name: for shape validation

    Returns:
      np.ndarray of shape (AUGMENTED_OBS_DIM[env_name],) — 23 or 26.

    Raises:
      ValueError if shapes don't match expected dimensions.
    """
    expected_raw = config.ENVIRONMENTS[env_name]['state_dim']
    if raw_vector.shape[0] != expected_raw:
        raise ValueError(
            f'Raw vector has {raw_vector.shape[0]} dims, '
            f'expected {expected_raw} for {env_name}'
        )

    bias = np.array(bias_vector, dtype=np.float32)
    if bias.shape[0] != config.BIAS_VECTOR_DIM:
        raise ValueError(
            f'Bias vector has {bias.shape[0]} dims, '
            f'expected {config.BIAS_VECTOR_DIM}'
        )

    # Clip bias to prevent memory from dominating policy
    bias = np.clip(bias, -config.BIAS_CLIP, config.BIAS_CLIP)

    augmented = np.concatenate([raw_vector, bias])

    from crossmill.training_config import STRATEGY_DIM
    expected_aug = config.AUGMENTED_OBS_DIM[env_name] - STRATEGY_DIM
    assert augmented.shape[0] == expected_aug, (
        f'Augmented vector has {augmented.shape[0]} dims, '
        f'expected {expected_aug}'
    )

    return augmented


def zero_bias() -> np.ndarray:
    """Return an all-zeros bias vector. Used when memory is explicitly disabled."""
    return np.zeros(config.BIAS_VECTOR_DIM, dtype=np.float32)


if __name__ == '__main__':
    # --- safenutri ---
    sn_obs = {f: 0.5 for f in SAFENUTRI_OBS_FIELDS}

    sn_vec = obs_to_vector('safenutri', sn_obs)
    assert sn_vec.shape == (15,), f'Expected (15,), got {sn_vec.shape}'

    sn_aug_zero = augment_observation(sn_vec, zero_bias(), 'safenutri')
    assert sn_aug_zero.shape == (23,), f'Expected (23,), got {sn_aug_zero.shape}'
    assert np.all(sn_aug_zero[-8:] == 0.0), 'Last 8 elements should be 0'

    sn_aug_bias = augment_observation(sn_vec, [0.1] * 8, 'safenutri')
    assert sn_aug_bias.shape == (23,), f'Expected (23,), got {sn_aug_bias.shape}'
    assert np.allclose(sn_aug_bias[-8:], 0.1), 'Last 8 elements should be 0.1'

    sn_aug_clip = augment_observation(sn_vec, [1.0] * 8, 'safenutri')
    assert sn_aug_clip.shape == (23,), f'Expected (23,), got {sn_aug_clip.shape}'
    assert np.allclose(sn_aug_clip[-8:], config.BIAS_CLIP), (
        f'Last 8 elements should be clipped to {config.BIAS_CLIP}'
    )

    # --- megaforge ---
    mf_obs = {f: 0.5 for f in MEGAFORGE_OBS_FIELDS}

    mf_vec = obs_to_vector('megaforge', mf_obs)
    assert mf_vec.shape == (18,), f'Expected (18,), got {mf_vec.shape}'

    mf_aug = augment_observation(mf_vec, zero_bias(), 'megaforge')
    assert mf_aug.shape == (26,), f'Expected (26,), got {mf_aug.shape}'

    print('Augmentation OK: safenutri 15->23, megaforge 18->26')
