import os
from huggingface_hub import HfApi, create_repo


def push_artifacts_to_hub(repo_id: str, model_zip_path: str,
                          curve_png_path: str, summary: dict) -> None:
    """
    Push trained model artifacts to HuggingFace Hub.

    Creates the repo if it does not exist (exist_ok=True).
    Uploads three files:
      - model.zip         (trained RecurrentPPO checkpoint)
      - reward_curve.png  (training reward curve)
      - README.md         (auto-generated model card)

    Args:
      repo_id:         HF repo ID, e.g. 'username/crossmill-safenutri-easy'
      model_zip_path:  Path to the saved model zip file
      curve_png_path:  Path to the reward curve PNG
      summary:         The complete summary dict from the training script.
                       Must contain: env, task_id, memory_mode, timesteps,
                       seed, pre_score, post_score, delta,
                       safety_violation_rate, catastrophic_rate.
                       Optionally: mean_vit_c_retention (SafeNutri),
                       mean_carbon_error_pct, mean_coke_rate_kgpt (MegaForge).
    """
    api = HfApi()
    create_repo(repo_id, repo_type='model', exist_ok=True)

    card = _build_model_card(repo_id, summary, model_zip_path)

    card_path = os.path.join(os.path.dirname(model_zip_path), 'README.md')
    with open(card_path, 'w') as f:
        f.write(card)

    files_to_upload = [
        (model_zip_path, 'model.zip'),
        (curve_png_path, 'reward_curve.png'),
        (card_path,      'README.md'),
    ]
    for local_path, repo_filename in files_to_upload:
        if os.path.exists(local_path):
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_filename,
                repo_id=repo_id,
                repo_type='model',
            )
            print(f'  Uploaded: {repo_filename}')
        else:
            print(f'  WARNING: {local_path} not found, skipping')

    print(f'Push complete: https://huggingface.co/{repo_id}')


def _build_model_card(repo_id: str, summary: dict,
                      model_zip_path: str) -> str:
    """Generate the README.md model card string. Extracted so __main__ can test
    card generation without triggering any HF API calls."""
    env_display  = 'SafeNutri' if summary['env'] == 'safenutri' else 'MegaForge'
    mode_display = {
        'none':  'No Memory (baseline — no cross-industry transfer)',
        'local': 'Local Memory (same-industry experience replay)',
        'cross': 'Cross-Industry Transfer (full CrossIndustryMemory active)',
    }.get(summary['memory_mode'], summary['memory_mode'])

    # Build env-specific results rows
    env_rows = ''
    if summary['env'] == 'safenutri' and summary.get('mean_vit_c_retention') is not None:
        env_rows += f'| Mean Vitamin C retention | {summary["mean_vit_c_retention"]:.3f} |\n'
    if summary['env'] == 'megaforge':
        if summary.get('mean_carbon_error_pct') is not None:
            env_rows += f'| Mean carbon error | {summary["mean_carbon_error_pct"]:.3f}% |\n'
        if summary.get('mean_coke_rate_kgpt') is not None:
            env_rows += f'| Mean coke rate | {summary["mean_coke_rate_kgpt"]:.1f} kg/t |\n'

    return f"""# {repo_id}

**CrossMill {env_display} — {summary['task_id'].capitalize()} task**

RecurrentPPO (LSTM) policy trained on CrossMill-{env_display} as part of the
CrossMill cross-industry RL platform. CrossMill connects two POMDP industrial
environments (orange juice pasteurization and blast furnace steel production)
through a shared CrossIndustryMemory layer that enables knowledge transfer
between industries.

## Memory Mode

**{mode_display}**

## Results

| Metric | Value |
|--------|-------|
| Pre-training heuristic baseline | {summary['pre_score']:.3f} |
| Post-training grader score | {summary['post_score']:.3f} |
| Delta | {summary['delta']:+.3f} |
| Safety violation rate | {summary['safety_violation_rate']:.3f} |
| Catastrophic failure rate | {summary['catastrophic_rate']:.3f} |
{env_rows}| Training timesteps | {summary['timesteps']:,} |
| Seed | {summary['seed']} |

## Files

- `model.zip` — trained RecurrentPPO checkpoint (SB3 / sb3-contrib format)
- `reward_curve.png` — training reward curve with baseline reference

## Reproduce

```bash
git clone https://github.com/YOUR-USERNAME/crossmill-integration
cd crossmill-integration
pip install -r requirements.txt
python scripts/train.py \\
    --env {summary['env']} \\
    --task {summary['task_id']} \\
    --memory_mode {summary['memory_mode']} \\
    --timesteps {summary['timesteps']} \\
    --seed {summary['seed']}
```

## About CrossMill

CrossMill is a cross-industry reinforcement learning platform built for the
OpenEnv AI Hackathon 2026 (Meta x Scaler School of Technology). It demonstrates
that control knowledge learned in steel production can accelerate learning in
food pasteurization, and vice versa, through a shared episodic-semantic memory
architecture.
"""


if __name__ == '__main__':
    sample_summary = {
        'env':                    'safenutri',
        'task_id':                'easy',
        'memory_mode':            'cross',
        'timesteps':              100_000,
        'seed':                   42,
        'pre_score':              0.312,
        'post_score':             0.671,
        'delta':                  0.359,
        'mean_reward':            0.648,
        'std_reward':             0.112,
        'safety_violation_rate':  0.04,
        'catastrophic_rate':      0.00,
        'mean_vit_c_retention':   0.783,
        'mean_carbon_error_pct':  None,
        'mean_coke_rate_kgpt':    None,
        'mean_co2_emissions_kgpt': None,
        'monitor_csv':            './runs/safenutri/monitor.monitor.csv',
        'model_zip':              './runs/safenutri/safenutri-easy-cross-ppo.zip',
        'curve_png':              './runs/safenutri/reward_curve_easy_cross.png',
    }

    card = _build_model_card(
        repo_id='inimay05/crossmill-safenutri-easy',
        summary=sample_summary,
        model_zip_path=sample_summary['model_zip'],
    )

    print(card)

    # Verify required content
    assert 'safenutri' in card.lower(), 'card missing env name'
    assert 'Cross-Industry Transfer' in card, 'card missing memory mode'
    assert '| Pre-training heuristic baseline |' in card, 'card missing results table'
    assert '0.312' in card, 'card missing pre_score'
    assert '0.671' in card, 'card missing post_score'
    assert '+0.359' in card, 'card missing delta'
    assert 'Mean Vitamin C retention' in card, 'card missing vit_c row'

    print('hub_push OK: model card generation works')
