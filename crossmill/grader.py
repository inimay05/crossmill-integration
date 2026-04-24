import os
import sys
import json
import argparse

from crossmill.grader_config import (
    GRADER_TARGETS,
    REQUIRED_SUMMARY_FIELDS,
    OPTIONAL_QUALITY_FIELDS,
    ALL_CONDITIONS,
    REQUIRED_CONDITIONS,
    OPTIONAL_CONDITIONS,
    REPORT_JSON_TEMPLATE,
    COMPARISON_PNG_TEMPLATE,
    DELTA_LABELS,
)
from crossmill.grader_validation import validate_summary, ValidationResult
from crossmill.plotting import plot_comparison_curves


def load_summary(log_dir: str, task: str, mode: str) -> dict | None:
    """
    Load summary_{task}_{mode}.json from log_dir.
    Returns None if the file does not exist (condition not yet trained).
    Raises ValueError if the file exists but is missing required fields.
    """
    path = os.path.join(log_dir, f'summary_{task}_{mode}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        summary = json.load(f)
    missing = [k for k in REQUIRED_SUMMARY_FIELDS if k not in summary]
    if missing:
        raise ValueError(
            f"Summary JSON at {path} is missing required fields: {missing}\n"
            f"Re-run the training script to regenerate this summary JSON.\n"
            f"The updated training guide (v2) writes all required fields."
        )
    return summary


def compute_deltas(validations: dict[str, ValidationResult],
                   summaries: dict[str, dict]) -> dict:
    """
    Compute the three delta metrics using raw grader_score (not adjusted_score).

    Args:
      validations: dict mapping mode -> ValidationResult
      summaries: dict mapping mode -> raw summary dict

    Returns dict with:
      local_gain:    local_grader_score - none_grader_score (None if local missing)
      cross_gain:    cross_grader_score - none_grader_score
      transfer_gain: cross_grader_score - local_grader_score (None if local missing)

    All deltas use raw grader_score to avoid adjusted_score inflating or
    deflating the comparison due to flag penalties.
    """
    none_score  = summaries['none']['grader_score']  if 'none'  in summaries else None
    local_score = summaries['local']['grader_score'] if 'local' in summaries else None
    cross_score = summaries['cross']['grader_score'] if 'cross' in summaries else None

    deltas = {}
    if none_score is not None and local_score is not None:
        deltas['local_gain']    = round(local_score - none_score, 4)
    else:
        deltas['local_gain']    = None

    if none_score is not None and cross_score is not None:
        deltas['cross_gain']    = round(cross_score - none_score, 4)
    else:
        deltas['cross_gain']    = None

    if local_score is not None and cross_score is not None:
        deltas['transfer_gain'] = round(cross_score - local_score, 4)
    else:
        deltas['transfer_gain'] = None

    return deltas


def print_report(env: str, task: str,
                 summaries: dict[str, dict],
                 validations: dict[str, ValidationResult],
                 deltas: dict) -> None:
    """
    Print a formatted comparison table to the terminal.
    """
    width = 72
    print(f'\n{"="*width}')
    print(f'  CrossMill Unified Grader')
    print(f'  Environment: {"SafeNutri" if env == "safenutri" else "MegaForge"}  |  '
          f'Task: {task.capitalize()}')
    print(f'{"="*width}')

    header = f'{"Condition":<12}  {"grader_score":>12}  {"adjusted":>10}  '
    header += f'{"stability":>10}  {"flags"}'
    print(f'\n{header}')
    print('-' * width)

    for mode in ALL_CONDITIONS:
        if mode not in validations:
            print(f'  {mode:<10}  {"(not run)":<12}')
            continue
        v = validations[mode]
        flag_str = ', '.join(v.flags) if v.flags else '-'
        score_str   = f'{v.grader_score:.4f}'
        adj_str     = f'{v.adjusted_score:.4f}'
        stab_str    = f'{v.stability:.4f}'
        adj_marker = ' *' if v.score_was_modified else '  '
        print(f'  {mode:<10}  {score_str:>12}  {adj_str:>8}{adj_marker}  '
              f'{stab_str:>10}  {flag_str}')

    print('\n  (* adjusted score was modified by anti-hacking rules)')

    print(f'\n  Deltas (computed on raw grader_score):')
    for key, label in DELTA_LABELS.items():
        val = deltas.get(key)
        if val is None:
            print(f'    {label:<45} n/a (condition not run)')
        else:
            sign = '+' if val >= 0 else ''
            print(f'    {label:<45} {sign}{val:.4f}')

    any_flags = any(v.has_flags for v in validations.values())
    print(f'\n  Anti-hacking: {"FLAGS RAISED — see table above" if any_flags else "All conditions passed validation."}')

    verdicts = [v.verdict for v in validations.values()]
    if 'CATASTROPHIC' in verdicts:
        overall = 'CATASTROPHIC FAILURE DETECTED'
    elif 'WARN_COMPOUND' in verdicts:
        overall = 'COMPOUND FAILURE — score reduced'
    elif 'WARN' in verdicts:
        overall = 'WARNINGS raised — review flags'
    else:
        overall = 'PASS'
    print(f'  Verdict: {overall}')
    print(f'{"="*width}\n')


def run_grader(env: str, task: str, log_dir: str,
               output_dir: str | None = None) -> dict:
    """
    Main grader function. Loads summaries, validates, computes deltas,
    generates the comparison plot, writes the report JSON.

    Args:
      env:        'safenutri' or 'megaforge'
      task:       'easy', 'medium', or 'hard'
      log_dir:    directory containing summary JSON files
                  (e.g. './runs/safenutri/')
      output_dir: where to write the report JSON and comparison PNG.
                  Defaults to log_dir.

    Returns:
      The full report dict (same as what is written to JSON).
    """
    if output_dir is None:
        output_dir = log_dir
    os.makedirs(output_dir, exist_ok=True)

    summaries = {}
    for mode in ALL_CONDITIONS:
        s = load_summary(log_dir, task, mode)
        if s is not None:
            summaries[mode] = s

    for required in REQUIRED_CONDITIONS:
        if required not in summaries:
            raise FileNotFoundError(
                f"Required summary for mode='{required}' not found in {log_dir}.\n"
                f"Run: python scripts/train.py --env {env} --task {task} "
                f"--memory_mode {required}\n"
                f"Then re-run the grader."
            )

    validations = {}
    for mode, summary in summaries.items():
        validations[mode] = validate_summary(summary)

    deltas = compute_deltas(validations, summaries)

    print_report(env, task, summaries, validations, deltas)

    csv_paths = {}
    for mode, summary in summaries.items():
        csv_paths[mode] = summary.get('monitor_csv')

    comparison_png = os.path.join(
        output_dir, COMPARISON_PNG_TEMPLATE.format(env=env, task=task)
    )

    if any(p and os.path.exists(p) for p in csv_paths.values()):
        baseline_floor  = summaries.get('none', summaries['cross'])['pre_score']
        grader_target   = GRADER_TARGETS[env][task]
        plot_comparison_curves(
            csv_paths=csv_paths,
            out_png_path=comparison_png,
            env_name=env,
            task_id=task,
            baseline_floor=baseline_floor,
            grader_target=grader_target,
        )
    else:
        print('  NOTE: No monitor CSV paths found — comparison plot skipped.')
        print('  Plots are generated during training. Run training first if needed.')
        comparison_png = None

    report = {
        'env':  env,
        'task': task,
        'scores': {
            mode: {
                'grader_score':   v.grader_score,
                'adjusted_score': v.adjusted_score,
                'stability':      round(v.stability, 4),
                'flags':          v.flags,
                'verdict':        v.verdict,
            }
            for mode, v in validations.items()
        },
        'deltas': deltas,
        'anti_hack_summary': (
            'All conditions passed validation.'
            if not any(v.has_flags for v in validations.values())
            else 'Flags raised — see scores section.'
        ),
        'comparison_png': comparison_png,
    }

    report_path = os.path.join(
        output_dir, REPORT_JSON_TEMPLATE.format(env=env, task=task)
    )
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  Report saved: {report_path}')

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrossMill Unified Grader')
    parser.add_argument('--env', required=True,
                        choices=['safenutri', 'megaforge'])
    parser.add_argument('--task', default='easy',
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--log_dir', default=None,
                        help='Directory containing summary JSON files. '
                             'Default: ./runs/<env>/')
    parser.add_argument('--output_dir', default=None,
                        help='Where to write report JSON and comparison plot. '
                             'Default: same as log_dir.')
    parser.add_argument('--verify', action='store_true',
                        help='Run a 10-episode spot-check on each condition '
                             'to confirm summary JSON numbers are plausible. '
                             'Requires trained model zips to exist.')
    args = parser.parse_args()

    log_dir = args.log_dir or f'./runs/{args.env}/'
    output_dir = args.output_dir or log_dir

    if args.verify:
        print('NOTE: --verify spot-check is not yet implemented. '
              'Running standard grader without spot-check.')

    report = run_grader(
        env=args.env,
        task=args.task,
        log_dir=log_dir,
        output_dir=output_dir,
    )
    sys.exit(0 if report['anti_hack_summary'].startswith('All') else 1)
