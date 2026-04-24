import matplotlib
matplotlib.use('Agg')   # must come before pyplot
import matplotlib.pyplot as plt
import pandas as pd
import os
from crossmill.training_config import ROLLING_WINDOW, PLOT_DPI


def plot_reward_curve(csv_path: str, out_png_path: str,
                      env_name: str, task_id: str, memory_mode: str,
                      baseline_score: float, final_score: float,
                      window: int = ROLLING_WINDOW) -> None:
    """
    Read a single VecMonitor CSV and save a single reward curve PNG.

    VecMonitor CSV format:
      - First line: a comment starting with '#' (skip it)
      - Columns: r (episode reward), l (episode length), t (wall time)
    """
    df = pd.read_csv(csv_path, skiprows=1)
    df['rolling'] = df['r'].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df.index, df['r'],
            alpha=0.25, linewidth=0.8, color='steelblue',
            label='episode reward')

    ax.plot(df.index, df['rolling'],
            linewidth=2.0, color='steelblue',
            label=f'rolling mean (w={window})')

    ax.axhline(baseline_score, linestyle='--', color='gray', linewidth=1.5,
               label=f'heuristic baseline ({baseline_score:.3f})')

    ax.axhline(final_score, linestyle='--', color='green', linewidth=1.5,
               label=f'post-training grader score ({final_score:.3f})')

    env_display = 'SafeNutri' if env_name == 'safenutri' else 'MegaForge'
    mode_display = {
        'none':  'No Memory (baseline)',
        'local': 'Local Memory',
        'cross': 'Cross-Industry Transfer',
    }.get(memory_mode, memory_mode)

    ax.set_title(f'CrossMill {env_display} — {task_id.capitalize()} — {mode_display}',
                 fontsize=13)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Total Episode Reward', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f'Reward curve saved: {out_png_path}')


def plot_comparison_curves(csv_paths: dict, out_png_path: str,
                           env_name: str, task_id: str,
                           baseline_floor: float,
                           grader_target: float,
                           window: int = ROLLING_WINDOW) -> None:
    """
    Plot three reward curves (none / local / cross) on the same axes.
    This is the unified grader's demo-ready three-curve overlay plot.

    Args:
      csv_paths: dict with keys 'none', 'local', 'cross' mapping to
                 VecMonitor CSV file paths. Missing keys are skipped.
      out_png_path: where to save the PNG
      env_name: 'safenutri' or 'megaforge'
      task_id: 'easy', 'medium', or 'hard'
      baseline_floor: mean_reward from the heuristic baseline (grey line)
      grader_target: the task's grader_target score (green dashed line)
      window: rolling mean window
    """
    # Colours: red=none, orange=local, blue=cross
    CURVE_STYLES = {
        'none':  {'color': '#C0392B', 'label': 'No Memory (baseline)'},
        'local': {'color': '#E67E22', 'label': 'Local Memory'},
        'cross': {'color': '#2E86C1', 'label': 'Cross-Industry Transfer'},
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    for mode, csv_path in csv_paths.items():
        if not csv_path or not os.path.exists(csv_path):
            continue
        style = CURVE_STYLES.get(mode, {'color': 'grey', 'label': mode})
        df = pd.read_csv(csv_path, skiprows=1)
        df['rolling'] = df['r'].rolling(window=window, min_periods=1).mean()
        ax.plot(df.index, df['r'],
                alpha=0.15, linewidth=0.6, color=style['color'])
        ax.plot(df.index, df['rolling'],
                linewidth=2.2, color=style['color'], label=style['label'])

    ax.axhline(baseline_floor, linestyle=':', color='grey', linewidth=1.5,
               label=f'Heuristic baseline ({baseline_floor:.3f})')
    ax.axhline(grader_target, linestyle='--', color='#1E8449', linewidth=1.8,
               label=f'Grader target ({grader_target:.3f})')

    env_display = 'SafeNutri' if env_name == 'safenutri' else 'MegaForge'
    ax.set_title(f'CrossMill {env_display} — {task_id.capitalize()} — Three-Condition Comparison',
                 fontsize=13)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Total Episode Reward (Rolling Mean)', fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f'Comparison plot saved: {out_png_path}')


if __name__ == '__main__':
    import numpy as np

    # ---- Synthetic VecMonitor CSV ----
    csv_path = '/tmp/test_monitor.monitor.csv'
    rng = np.random.default_rng(0)
    n = 200
    r_vals = np.linspace(-0.5, 0.8, n) + rng.normal(0, 0.15, n)
    l_vals = rng.integers(50, 200, n)
    t_vals = np.cumsum(rng.uniform(0.5, 2.0, n))

    with open(csv_path, 'w') as f:
        f.write('#{"t_start": 0.0, "env_id": "CrossMillGymShim"}\n')
        f.write('r,l,t\n')
        for r, l, t in zip(r_vals, l_vals, t_vals):
            f.write(f'{r:.6f},{l},{t:.3f}\n')

    # ---- Single-curve plot ----
    single_png = '/tmp/test_single_curve.png'
    plot_reward_curve(
        csv_path=csv_path,
        out_png_path=single_png,
        env_name='safenutri',
        task_id='easy',
        memory_mode='cross',
        baseline_score=-0.2,
        final_score=0.7,
    )
    assert os.path.exists(single_png), 'single curve PNG not created'
    assert os.path.getsize(single_png) > 0, 'single curve PNG is empty'

    # ---- Comparison plot (same CSV for all three modes) ----
    comparison_png = '/tmp/test_comparison_curve.png'
    plot_comparison_curves(
        csv_paths={'none': csv_path, 'local': csv_path, 'cross': csv_path},
        out_png_path=comparison_png,
        env_name='safenutri',
        task_id='easy',
        baseline_floor=-0.2,
        grader_target=0.75,
    )
    assert os.path.exists(comparison_png), 'comparison PNG not created'
    assert os.path.getsize(comparison_png) > 0, 'comparison PNG is empty'

    print('Plotting OK: single curve and comparison curve both work')
