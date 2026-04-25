import base64
import glob
import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd


# ── helpers ────────────────────────────────────────────────

def _load_story_html() -> str:
    """Return an iframe embedding crossmill_story.html via base64 data URI."""
    path = Path("crossmill_story.html")
    if not path.exists():
        return (
            '<div style="padding:2rem;text-align:center;color:#7F8C9B;">'
            "<p>crossmill_story.html not found in Space root.</p>"
            "</div>"
        )
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    return (
        f'<iframe src="data:text/html;base64,{b64}" '
        'style="width:100%;height:920px;border:none;" '
        "frameborder='0'></iframe>"
    )


def _normalise_path(raw: str) -> str:
    """Convert Windows-style backslash paths written on-device to forward slashes."""
    return raw.replace("\\", "/")


def _load_results() -> tuple[pd.DataFrame | None, list[str]]:
    """
    Scan runs/*/summary_*.json and return (DataFrame, list_of_curve_paths).
    Returns (None, []) when no summary files exist yet.
    """
    summary_files = sorted(glob.glob("runs/*/summary_*.json"))
    if not summary_files:
        return None, []

    rows: list[dict] = []
    curves: list[str] = []

    for fpath in summary_files:
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        pre   = d.get("pre_score")
        post  = d.get("post_score")
        delta = d.get("delta")

        if pre is not None and pre != 0 and delta is not None:
            tg = f"{(delta / abs(pre)) * 100:+.1f}%"
        else:
            tg = "N/A"

        rows.append(
            {
                "Environment":   d.get("env", "—").title(),
                "Task":          d.get("task_id", "—").title(),
                "Mode":          d.get("memory_mode", "—"),
                "Timesteps":     d.get("timesteps", "—"),
                "Pre Score":     f"{pre:.4f}"   if pre   is not None else "—",
                "Post Score":    f"{post:.4f}"  if post  is not None else "—",
                "Delta":         f"{delta:+.4f}" if delta is not None else "—",
                "Transfer Gain": tg,
                "Mean Reward":   f"{d['mean_reward']:.4f}" if d.get("mean_reward") is not None else "—",
            }
        )

        raw_curve = d.get("curve_png", "")
        if raw_curve:
            curve = _normalise_path(raw_curve)
            if os.path.exists(curve):
                curves.append(curve)

    if not rows:
        return None, []

    return pd.DataFrame(rows), curves


# ── load once at startup ────────────────────────────────────
_STORY_HTML  = _load_story_html()
_RESULTS_DF, _CURVES = _load_results()

_RESULTS_PLACEHOLDER = (
    '<div style="padding:2.5rem;text-align:center;'
    'font-family:Inter,system-ui,sans-serif;color:#7F8C9B;">'
    '<p style="font-size:1.2rem;margin-bottom:0.6rem;">⏳ Training in progress</p>'
    '<p style="font-size:0.85rem;line-height:1.7;">'
    "Results will appear here once <code>summary_*.json</code> files "
    "are written to the <code>runs/</code> directory after training completes."
    "</p>"
    "</div>"
)

# ── Gradio app ──────────────────────────────────────────────

_CSS = """
.gradio-container { max-width: 1280px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="CrossMill — Cross-Industry RL Platform") as demo:

    gr.HTML(
        """
        <div style="padding:1.2rem 0 0.2rem;text-align:center;
                    font-family:Inter,system-ui,sans-serif;">
          <h1 style="font-size:1.9rem;font-weight:900;margin:0;
                     background:linear-gradient(135deg,#fff 40%,#5DADE2 100%);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            CrossMill
          </h1>
          <p style="color:#7F8C9B;margin:0.3rem 0 0;font-size:0.85rem;">
            Cross-Industry Reinforcement Learning &nbsp;·&nbsp;
            OpenEnv AI Hackathon 2026 &nbsp;·&nbsp;
            Meta × Scaler School of Technology
          </p>
        </div>
        """
    )

    with gr.Tabs():

        # ── Tab 1: story page ──────────────────────────────
        with gr.Tab("🏭  CrossMill Story"):
            gr.HTML(_STORY_HTML)

        # ── Tab 2: results dashboard ───────────────────────
        with gr.Tab("📊  Results Dashboard"):

            if _RESULTS_DF is None:
                gr.HTML(_RESULTS_PLACEHOLDER)

            else:
                gr.HTML(
                    '<h3 style="font-family:Inter,sans-serif;'
                    'margin:1rem 0 0.4rem;font-size:1rem;">'
                    "Training Results</h3>"
                )
                gr.Dataframe(
                    value=_RESULTS_DF,
                    label=None,
                    interactive=False,
                    wrap=True,
                )

                if _CURVES:
                    gr.HTML(
                        '<h3 style="font-family:Inter,sans-serif;'
                        'margin:1.8rem 0 0.4rem;font-size:1rem;">'
                        "Reward Curves</h3>"
                    )
                    gr.Gallery(
                        value=_CURVES,
                        label="Reward curves",
                        columns=2,
                        height="auto",
                    )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css=_CSS,
    )
