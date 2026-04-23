from typing import Any
from crossmill.memory.config import ACTION_PATTERNS


def _coerce_safenutri_action(action: Any) -> dict:
    """
    Normalise a SafeNutri action into a dict with the 8 named fields.
    Accepts: dict, pydantic model with model_dump, or list/numpy array of len 8.
    Field order for arrays:
      [target_temp, heating_rate, hold_time, cooling_rate, flow_adjust,
       n_stages, checkpoint_interval, emergency_stop]
    All fields are normalised [0,1] except n_stages (int 2-5) and
    emergency_stop (0 or 1).
    """
    if hasattr(action, 'model_dump'):
        return action.model_dump()
    if isinstance(action, dict):
        return action
    # Assume list-like
    fields = ['target_temp', 'heating_rate', 'hold_time', 'cooling_rate',
              'flow_adjust', 'n_stages', 'checkpoint_interval', 'emergency_stop']
    vals = list(action)
    if len(vals) != 8:
        raise ValueError(f"SafeNutri action length {len(vals)}, expected 8")
    return {k: float(v) for k, v in zip(fields, vals)}


def _coerce_megaforge_action(action: Any) -> dict:
    """
    Normalise a MegaForge action into a dict with the 10 named fields.
    Field order for arrays:
      [oxygen_flow_delta, blast_temp, coke_feed_delta, ore_feed_delta,
       limestone_addition, tapping_interval, temp_ramp_rate, pressure_target,
       alloy_timing, emergency_cooling]
    """
    if hasattr(action, 'model_dump'):
        return action.model_dump()
    if isinstance(action, dict):
        return action
    fields = ['oxygen_flow_delta', 'blast_temp', 'coke_feed_delta',
              'ore_feed_delta', 'limestone_addition', 'tapping_interval',
              'temp_ramp_rate', 'pressure_target', 'alloy_timing',
              'emergency_cooling']
    vals = list(action)
    if len(vals) != 10:
        raise ValueError(f"MegaForge action length {len(vals)}, expected 10")
    return {k: float(v) for k, v in zip(fields, vals)}


def classify_safenutri_action(action: Any) -> str:
    """
    Classify a SafeNutri action into one of the 5 ACTION_PATTERNS labels.

    Rules (first match wins):
      1. emergency_response   : emergency_stop >= 0.5
      2. rapid_correction     : heating_rate > 0.75 OR cooling_rate > 0.67
                                (heating > 6.0 °C/s mapped to normalised >0.75,
                                 cooling > 10 °C/s mapped to normalised >0.67)
      3. gradual_ramp         : heating_rate < 0.31 AND hold_time > 0.22
                                (heating < 3.0 °C/s is <31% of [0.5, 8.0] range,
                                 hold_time > 30 s is >22% of [5, 120] range)
      4. hold_steady          : |flow_adjust - 0.5| < 0.10 AND
                                heating_rate between 0.30 and 0.55
                                (flow_adjust near centre = no flow change)
      5. cautious_explore     : fallback for everything else — small non-zero
                                adjustments without matching a stronger pattern
    """
    d = _coerce_safenutri_action(action)
    if float(d.get('emergency_stop', 0)) >= 0.5:
        return 'emergency_response'
    heating = float(d.get('heating_rate', 0.5))
    cooling = float(d.get('cooling_rate', 0.5))
    hold = float(d.get('hold_time', 0.5))
    flow_adj = float(d.get('flow_adjust', 0.5))
    if heating > 0.75 or cooling > 0.67:
        return 'rapid_correction'
    if heating < 0.31 and hold > 0.22:
        return 'gradual_ramp'
    if abs(flow_adj - 0.5) < 0.10 and 0.30 <= heating <= 0.55:
        return 'hold_steady'
    return 'cautious_explore'


def classify_megaforge_action(action: Any) -> str:
    """
    Classify a MegaForge action into one of the 5 ACTION_PATTERNS labels.

    Rules (first match wins):
      1. emergency_response : emergency_cooling >= 0.5
      2. rapid_correction   : temp_ramp_rate > 0.67 OR
                              |oxygen_flow_delta - 0.5| > 0.42
                              (ramp > 35 °C/h is >67% of [5,50] range;
                               oxygen delta beyond ±25% means >42% from centre)
      3. gradual_ramp       : temp_ramp_rate < 0.33 AND
                              |coke_feed_delta - 0.5| < 0.25
                              (ramp < 20 °C/h is <33%; coke delta within ±10%
                               of centre means <25% from 0.5)
      4. hold_steady        : |oxygen_flow_delta - 0.5| < 0.17 AND
                              |ore_feed_delta - 0.5| < 0.17
                              (oxygen delta within ±10%, ore within ±5%)
      5. cautious_explore   : fallback
    """
    d = _coerce_megaforge_action(action)
    if float(d.get('emergency_cooling', 0)) >= 0.5:
        return 'emergency_response'
    ramp = float(d.get('temp_ramp_rate', 0.5))
    ox = float(d.get('oxygen_flow_delta', 0.5))
    coke = float(d.get('coke_feed_delta', 0.5))
    ore = float(d.get('ore_feed_delta', 0.5))
    if ramp > 0.67 or abs(ox - 0.5) > 0.42:
        return 'rapid_correction'
    if ramp < 0.33 and abs(coke - 0.5) < 0.25:
        return 'gradual_ramp'
    if abs(ox - 0.5) < 0.17 and abs(ore - 0.5) < 0.17:
        return 'hold_steady'
    return 'cautious_explore'


# ---- DISPATCHER ----
CLASSIFIERS = {
    'safenutri': classify_safenutri_action,
    'megaforge': classify_megaforge_action,
}


def classify_action(env_name: str, action: Any) -> str:
    if env_name not in CLASSIFIERS:
        raise ValueError(f"Unknown env_name: {env_name!r}")
    label = CLASSIFIERS[env_name](action)
    if label not in ACTION_PATTERNS:
        raise RuntimeError(f"Classifier returned invalid label {label!r}. "
                           f"Expected one of {ACTION_PATTERNS}")
    return label


if __name__ == '__main__':
    PASS = 0

    def check(env, action, expected):
        global PASS
        got = classify_action(env, action)
        status = 'OK' if got == expected else 'FAIL'
        print(f"  [{status}] {env:<10s}  expected={expected:<20s}  got={got}")
        assert got == expected, f"MISMATCH: expected {expected!r}, got {got!r}"
        PASS += 1

    print("SafeNutri:")
    check('safenutri',
          {'emergency_stop': 1.0},
          'emergency_response')
    check('safenutri',
          {'heating_rate': 0.9},
          'rapid_correction')
    check('safenutri',
          {'heating_rate': 0.2, 'hold_time': 0.5},
          'gradual_ramp')
    check('safenutri',
          {'heating_rate': 0.45, 'flow_adjust': 0.5},
          'hold_steady')
    check('safenutri',
          {'heating_rate': 0.60, 'flow_adjust': 0.62, 'cooling_rate': 0.55},
          'cautious_explore')

    print("\nMegaForge:")
    check('megaforge',
          {'emergency_cooling': 1.0},
          'emergency_response')
    check('megaforge',
          {'temp_ramp_rate': 0.8},
          'rapid_correction')
    check('megaforge',
          {'temp_ramp_rate': 0.2, 'coke_feed_delta': 0.5},
          'gradual_ramp')
    check('megaforge',
          {'oxygen_flow_delta': 0.5, 'ore_feed_delta': 0.5},
          'hold_steady')
    # ox=0.70 → |0.70-0.5|=0.20, outside hold_steady band (≥0.17) but
    # not beyond rapid_correction threshold (≤0.42); ramp=0.50 is mid-range.
    check('megaforge',
          {'temp_ramp_rate': 0.50, 'oxygen_flow_delta': 0.70,
           'coke_feed_delta': 0.65, 'ore_feed_delta': 0.65},
          'cautious_explore')

    print("\nDispatcher test:")
    label = classify_action('safenutri', {'emergency_stop': 0.0, 'heating_rate': 0.45,
                                          'flow_adjust': 0.5})
    print(f"  classify_action('safenutri', dict)  -> {label!r}")
    assert label in ACTION_PATTERNS

    print("\nArray input test:")
    label = classify_action('safenutri', [0.5] * 8)
    print(f"  classify_action('safenutri', [0.5]*8) -> {label!r}")
    assert label in ACTION_PATTERNS

    print(f"\nClassification OK: all {PASS} pattern mappings verified")
