"""
run_correction_vis.py
=====================
Run the full correction algorithm (Stages 1–5) on a single chart image
and produce an annotated 3-panel action figure for every iteration,
a convergence curve, and a final before/after comparison.

After launch the script prompts you to select which symbol types are
present in the chart.  Only the selected classes are used for detection,
NMS, and correction — all others are ignored.

Usage
-----
  # Basic — prompts for symbol selection interactively
  python run_correction_vis.py path/to/chart.png

  # Pre-select symbols via command line (space-separated 1-based indices)
  python run_correction_vis.py path/to/chart.png --classes 1 2 3 4

  # Custom output directory
  python run_correction_vis.py path/to/chart.png --out-dir path/to/output

  # Custom model path
  python run_correction_vis.py path/to/chart.png --model path/to/chart_marker_net_v3.pth

  # Change iteration cap (default 25)
  python run_correction_vis.py path/to/chart.png --t-max 15

  # Change sliding-window stride for Stage 1 (default 8, larger = faster)
  python run_correction_vis.py path/to/chart.png --stride 16

Output files (all inside --out-dir)
------------------------------------
  iter01.jpg … iterNN.jpg    Annotated 3-panel figure per iteration
  convergence.jpg             1-SSIM convergence curve
  final_before_after.jpg      Side-by-side original / before / after comparison
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── All symbol classes (indices 1-based for user display) ────────────────────
ALL_SYMBOL_NAMES = [
    "filled_circle",        # 1
    "open_circle",          # 2
    "filled_square",        # 3
    "open_square",          # 4
    "open_triangle",        # 5
    "open_inv_triangle",    # 6
    "filled_triangle",      # 7
    "filled_inv_triangle",  # 8
    "open_rhombus",         # 9
    "filled_rhombus",       # 10
    "x_marker",             # 11
    "plus_marker",          # 12
]

# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="Run correction algorithm with per-iteration visualisation")
    p.add_argument("image", type=Path,
                   help="Path to the input chart image (.png / .jpg)")
    p.add_argument("--model", type=Path, default=None,
                   help="Path to chart_marker_net_v3.pth  "
                        "(default: ../models/chart_marker_net_v3.pth relative to this script)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory  "
                        "(default: ../experiments/correction_vis/<image_stem>)")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="1-based indices of symbol classes present in the chart "
                        "(e.g. --classes 1 2 3 4).  If omitted, the script prompts "
                        "interactively.")
    p.add_argument("--t-max", type=int, default=25,
                   help="Maximum number of correction iterations  (default: 25)")
    p.add_argument("--stride", type=int, default=8,
                   help="Sliding-window stride for Stage 1 detection  (default: 8)")
    return p.parse_args()


# ── Interactive symbol-selection prompt ───────────────────────────────────────
def _prompt_symbol_selection():
    """Print the numbered symbol menu and return the selected class names."""
    print()
    print("=" * 60)
    print("  Symbol classes present in this chart")
    print("=" * 60)
    for i, name in enumerate(ALL_SYMBOL_NAMES, start=1):
        print(f"  {i:2d}.  {name}")
    print("=" * 60)
    print("  Enter the numbers of the symbols present in the chart,")
    print("  separated by spaces.  Example:  1 2 3 4")
    print("  (Press Enter with no input to use ALL 12 classes.)")
    print()

    while True:
        raw = input("  Your selection: ").strip()
        if raw == "":
            selected = list(ALL_SYMBOL_NAMES)
            print(f"\n  Using all {len(selected)} symbol classes.\n")
            return selected
        try:
            indices = [int(x) for x in raw.split()]
        except ValueError:
            print("  Invalid input — please enter space-separated integers.")
            continue
        invalid = [x for x in indices if x < 1 or x > len(ALL_SYMBOL_NAMES)]
        if invalid:
            print(f"  Out-of-range values: {invalid}. "
                  f"Valid range is 1–{len(ALL_SYMBOL_NAMES)}.")
            continue
        selected = [ALL_SYMBOL_NAMES[i - 1] for i in sorted(set(indices))]
        print()
        print("  Selected classes:")
        for name in selected:
            print(f"    • {name}")
        print()
        return selected


# ── Module loader ─────────────────────────────────────────────────────────────
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


# ── Class colour map ──────────────────────────────────────────────────────────
CLASS_COLORS = {
    'filled_circle':       '#0077BB',
    'open_circle':         '#EE7733',
    'filled_square':       '#009944',
    'open_square':         '#CC3311',
    'open_triangle':       '#33BBEE',
    'open_inv_triangle':   '#EE3377',
    'filled_triangle':     '#BBBBBB',
    'filled_inv_triangle': '#44BB99',
    'open_rhombus':        '#DDAA33',
    'filled_rhombus':      '#AA3377',
    'x_marker':            '#004488',
    'plus_marker':         '#994455',
}


# ── Per-iteration annotated figure ───────────────────────────────────────────
def draw_iter_fig(t, ssim_before, ssim_after, state, I0, I_before, I_after,
                  PT_TOL, out_dir):
    action      = state["action"]
    l_star      = state["l_star"]
    anchor_l    = state["anchor_left"]
    anchor_r    = state["anchor_right"]
    added_pts   = state["added"]
    removed_pts = state["removed"]
    P_before    = state["P_in"]
    P_after     = state["P_out"]
    converged   = state["converged"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor('#F8F8F8')

    # ── Panel 0: Original ────────────────────────────────────────────────────
    axes[0].imshow(cv2.cvtColor(I0, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original chart  $I_0$", fontsize=13, fontweight='bold', pad=10)
    axes[0].axis('off')

    # ── Panel 1: Before ──────────────────────────────────────────────────────
    ax1 = axes[1]
    ax1.imshow(cv2.cvtColor(I_before, cv2.COLOR_BGR2RGB))
    if converged:
        ax1.set_title(f"Before iteration {t}  (1−SSIM = {ssim_before:.4f})\n"
                      "CONVERGED — no improvement found", fontsize=11, pad=10)
    else:
        ax1.set_title(
            f"Before iteration {t}  (1−SSIM = {ssim_before:.4f})\n"
            f"l* shown in red  |  search radius = {PT_TOL:.0f} px",
            fontsize=11, pad=10)
    ax1.axis('off')

    for pt in P_before:
        col = CLASS_COLORS.get(pt.get('class_name', ''), '#888888')
        ax1.plot(pt['cx'], pt['cy'], 'o', color=col, markersize=6,
                 markeredgecolor='white', markeredgewidth=0.8, zorder=4)

    if l_star is not None:
        lx1, ly1, lx2, ly2 = l_star
        ax1.plot([lx1, lx2], [ly1, ly2], '-', color='red', linewidth=3, zorder=5)
        for ap in [anchor_l, anchor_r]:
            ax1.plot(ap[0], ap[1], 'o', color='red', markersize=9, zorder=6)
            circ = plt.Circle(ap, PT_TOL, color='darkorange',
                              fill=False, linewidth=1.5, linestyle='--', zorder=5)
            ax1.add_patch(circ)

    if action == 'ACTIVATE':
        for pt in added_pts:
            cx, cy = pt['cx'], pt['cy']
            cn = pt.get('class_name', '')
            ax1.plot(cx, cy, '*', color='magenta', markersize=22,
                     markeredgecolor='black', markeredgewidth=0.8, zorder=8)
            ax1.annotate(
                f"  ACTIVATE\n  class: {cn}\n  ({cx:.0f}, {cy:.0f})",
                xy=(cx, cy), xytext=(cx + 18, cy - 18), fontsize=8.5,
                color='magenta', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='magenta', alpha=0.85),
                zorder=9)

    elif action == 'DELETE':
        for pt in removed_pts:
            cx, cy = pt['cx'], pt['cy']
            cn = pt.get('class_name', '')
            ax1.plot(cx, cy, 'x', color='red', markersize=18, markeredgewidth=3, zorder=8)
            ax1.annotate(
                f"  DELETE\n  class: {cn}\n  ({cx:.0f}, {cy:.0f})",
                xy=(cx, cy), xytext=(cx + 18, cy - 18), fontsize=8.5,
                color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.85),
                zorder=9)

    elif action == 'REPLACE':
        for pt in removed_pts:
            cx, cy = pt['cx'], pt['cy']
            cn = pt.get('class_name', '')
            ax1.plot(cx, cy, 'x', color='red', markersize=18, markeredgewidth=3, zorder=8)
            ax1.annotate(
                f"  REMOVE\n  class: {cn}\n  ({cx:.0f}, {cy:.0f})",
                xy=(cx, cy), xytext=(cx - 70, cy - 20), fontsize=8.5,
                color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.85),
                zorder=9)
        for pt in added_pts:
            cx, cy = pt['cx'], pt['cy']
            cn = pt.get('class_name', '')
            ax1.plot(cx, cy, '*', color='blue', markersize=22,
                     markeredgecolor='black', markeredgewidth=0.8, zorder=8)
            ax1.annotate(
                f"  ADD\n  class: {cn}\n  ({cx:.0f}, {cy:.0f})",
                xy=(cx, cy), xytext=(cx + 18, cy - 18), fontsize=8.5,
                color='blue', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='blue', alpha=0.85),
                zorder=9)

    # ── Panel 2: After ───────────────────────────────────────────────────────
    ax2 = axes[2]
    ax2.imshow(cv2.cvtColor(I_after, cv2.COLOR_BGR2RGB))
    if converged:
        ax2.set_title(f"After iteration {t}  (1−SSIM = {ssim_after:.4f})\n"
                      "No change — algorithm stopped", fontsize=11, pad=10)
    else:
        ax2.set_title(
            f"After iteration {t}  (1−SSIM = {ssim_after:.4f})\n"
            f"Action: {action}  |  Δ(1−SSIM) = {ssim_before - ssim_after:.4f}",
            fontsize=11, pad=10)
    ax2.axis('off')

    for pt in P_after:
        col = CLASS_COLORS.get(pt.get('class_name', ''), '#888888')
        ax2.plot(pt['cx'], pt['cy'], 'o', color=col, markersize=6,
                 markeredgecolor='white', markeredgewidth=0.8, zorder=4)

    for pt in added_pts:
        cx, cy = pt['cx'], pt['cy']
        col = CLASS_COLORS.get(pt.get('class_name', ''), '#888888')
        ax2.plot(cx, cy, 'o', color=col, markersize=14,
                 markeredgecolor='magenta', markeredgewidth=2.5, zorder=8)
        ax2.annotate(
            f"  Activated\n  ({cx:.0f}, {cy:.0f})",
            xy=(cx, cy), xytext=(cx + 18, cy - 18), fontsize=8.5,
            color='magenta', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='magenta', alpha=0.85),
            zorder=9)
    for pt in removed_pts:
        cx, cy = pt['cx'], pt['cy']
        ax2.plot(cx, cy, 'x', color='red', markersize=14, markeredgewidth=3, zorder=8)
        ax2.annotate(
            f"  Removed\n  ({cx:.0f}, {cy:.0f})",
            xy=(cx, cy), xytext=(cx + 18, cy + 10), fontsize=8.5,
            color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.85),
            zorder=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mlines.Line2D([0], [0], color='red', linewidth=3,
                      label='l* guiding segment'),
        mlines.Line2D([0], [0], color='darkorange', linewidth=1.5, linestyle='--',
                      label=f'Search radius ({PT_TOL:.0f} px)'),
        mlines.Line2D([0], [0], marker='*', color='magenta', markersize=14,
                      linestyle='None', markeredgecolor='black',
                      label='Activated (suppressed → active)'),
        mlines.Line2D([0], [0], marker='x', color='red', markersize=12,
                      linestyle='None', markeredgewidth=2.5,
                      label='Deleted / removed'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=10, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    title_str = (f"Correction Iteration {t}  |  Action: {action}  |  "
                 f"1−SSIM: {ssim_before:.4f} → {ssim_after:.4f}  "
                 f"(Δ = {ssim_before - ssim_after:.4f})")
    if converged:
        title_str = f"Correction Iteration {t}  |  CONVERGED (no improvement)"
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out_path = out_dir / f"iter{t:02d}.jpg"
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()

    SRC_DIR = Path(__file__).parent.resolve()

    # Resolve model path
    model_path = args.model
    if model_path is None:
        model_path = SRC_DIR.parent / "models" / "chart_marker_net_v3.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    # ── Symbol selection ─────────────────────────────────────────────────────
    if args.classes is not None:
        # Provided via --classes flag
        invalid = [x for x in args.classes if x < 1 or x > len(ALL_SYMBOL_NAMES)]
        if invalid:
            raise ValueError(f"--classes indices out of range (1–{len(ALL_SYMBOL_NAMES)}): "
                             f"{invalid}")
        selected_classes = [ALL_SYMBOL_NAMES[i - 1]
                            for i in sorted(set(args.classes))]
        print(f"\nUsing {len(selected_classes)} symbol class(es) from --classes flag:")
        for name in selected_classes:
            print(f"  • {name}")
        print()
    else:
        # Interactive prompt
        selected_classes = _prompt_symbol_selection()

    if not selected_classes:
        raise ValueError("No symbol classes selected — cannot run.")

    # Resolve output directory
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (SRC_DIR.parent / "experiments" / "correction_vis"
                   / args.image.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    DETECTOR_PY = SRC_DIR / "1_point_detection_v3.py"
    T_MAX  = args.t_max
    STRIDE = args.stride

    # ── Load modules ─────────────────────────────────────────────────────────
    print("Loading pipeline modules …")
    _load("chart_marker_detector_v3", DETECTOR_PY)
    mod2 = _load("adaptive_nms",     SRC_DIR / "2_point_detection_adaptive_nms.py")
    mod3 = _load("segment_detector", SRC_DIR / "3_segment_detection.py")
    mod4 = _load("refine_segments",  SRC_DIR / "4_segment_refinement.py")
    mod5 = _load("correction",       SRC_DIR / "5_correction.py")

    # ── Stages 1–3 ───────────────────────────────────────────────────────────
    print(f"Stage 1 — detection + NMS  (stride={STRIDE}, "
          f"classes={len(selected_classes)}) …")
    result2 = mod2.detect_with_adaptive_nms(
        img_path         = str(args.image),
        model_path       = str(model_path),
        known_classes    = selected_classes,          # ← constrained class set
        detector_py_path = str(DETECTOR_PY),
        stride           = STRIDE)
    kept_pts = result2["kept"]
    supp_pts = result2["suppressed"]
    mode_xs  = result2["mode_xs"]
    img_bgr  = result2["img_bgr"]
    grid_xs  = sorted(float(x) for x in mode_xs)
    print(f"  Kept: {len(kept_pts)}  Suppressed: {len(supp_pts)}  "
          f"Grid cols: {len(grid_xs)}")

    print("Stage 2 — segment detection …")
    segs_raw = mod3.detect(img_bgr)
    print(f"  Raw segments: {len(segs_raw)}")

    print("Stage 3 — segment refinement (cut-only) …")
    segs_ref, ref_log = mod4.refine(segs_raw, grid_xs)
    print(f"  Refined segments: {len(segs_ref)}  "
          f"(pruned={ref_log.get('n_pruned_short', 0)}, "
          f"cuts={ref_log.get('n_cuts', 0)})")

    H, W = img_bgr.shape[:2]
    grid_step  = mod4.grid_step(grid_xs)
    nms_window = grid_step * 0.75
    mod5.X_APPROX = grid_step * 0.5

    I0     = img_bgr
    PT_TOL = mod5.PT_TOL

    # ── Correction loop ───────────────────────────────────────────────────────
    P_cur = kept_pts
    S_cur = supp_pts
    ssim_history = []

    I_init   = mod5.render_from_points(P_cur, (H, W))
    ssim_cur = mod5.ssim_dist(I0, I_init)
    ssim_history.append(ssim_cur)
    print(f"\nStage 4 — correction  (T_MAX={T_MAX}, "
          f"classes={len(selected_classes)})")
    print(f"  Initial 1-SSIM: {ssim_cur:.5f}\n")

    iter_paths = []
    for t in range(1, T_MAX + 1):
        t0 = time.time()
        state = mod5.run_one_iteration(
            P_cur, S_cur, I0, segs_ref,
            selected_classes,                         # ← constrained class set
            nms_window=nms_window)
        elapsed = time.time() - t0

        ssim_before = ssim_cur
        ssim_after  = state["best_dist"] if state["improved"] else ssim_cur

        I_before = mod5.render_from_points(P_cur,          (H, W))
        I_after  = mod5.render_from_points(state["P_out"], (H, W))

        action_str = state["action"] if not state["converged"] else "CONVERGED"
        print(f"  Iter {t:2d}: {action_str:10s}  "
              f"added={len(state['added'])}  removed={len(state['removed'])}  "
              f"1-SSIM: {ssim_before:.5f} → {ssim_after:.5f}  "
              f"(Δ={ssim_before - ssim_after:.5f})  {elapsed:.1f}s")

        path = draw_iter_fig(t, ssim_before, ssim_after, state,
                             I0, I_before, I_after, PT_TOL, out_dir)
        iter_paths.append(path)

        ssim_history.append(ssim_after)
        P_cur = state["P_out"]
        S_cur = state["S_out"]

        if state["converged"] or not state["improved"]:
            print(f"\n  Stopped at iteration {t} (converged / no improvement).")
            break

    n_iters = len(iter_paths)

    # ── Convergence curve ─────────────────────────────────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(max(9, n_iters * 0.5), 4))
    ax_c.plot(range(n_iters + 1), ssim_history, 'o-', color='steelblue',
              linewidth=2, markersize=7)
    ax_c.set_xlabel("Iteration", fontsize=12)
    ax_c.set_ylabel("1 − SSIM", fontsize=12)
    ax_c.set_title("Correction Convergence Curve", fontsize=13, fontweight='bold')
    ax_c.set_xticks(range(n_iters + 1))
    ax_c.set_xticklabels(['init'] + [str(i + 1) for i in range(n_iters)])
    ax_c.grid(True, alpha=0.4)
    for i, v in enumerate(ssim_history):
        ax_c.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                      xytext=(0, 8), ha='center', fontsize=8)
    plt.tight_layout()
    conv_path = out_dir / "convergence.jpg"
    plt.savefig(str(conv_path), dpi=130, bbox_inches='tight')
    plt.close()

    # ── Final before/after ────────────────────────────────────────────────────
    I_final = mod5.render_from_points(P_cur, (H, W))
    fig_f, axes_f = plt.subplots(1, 3, figsize=(21, 7))
    fig_f.patch.set_facecolor('#F8F8F8')
    for ax, im, title in zip(
        axes_f,
        [cv2.cvtColor(I0,      cv2.COLOR_BGR2RGB),
         cv2.cvtColor(I_init,  cv2.COLOR_BGR2RGB),
         cv2.cvtColor(I_final, cv2.COLOR_BGR2RGB)],
        ["Original  $I_0$",
         f"Before correction  (1−SSIM = {ssim_history[0]:.4f})",
         f"After {n_iters} iterations  (1−SSIM = {ssim_history[-1]:.4f})"]):
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(title, fontsize=12, pad=8)
    fig_f.suptitle(
        f"Full Correction Result  |  "
        f"1−SSIM: {ssim_history[0]:.4f} → {ssim_history[-1]:.4f}  "
        f"(total Δ = {ssim_history[0] - ssim_history[-1]:.4f}  "
        f"over {n_iters} iterations)  |  "
        f"classes: {', '.join(selected_classes)}",
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    final_path = out_dir / "final_before_after.jpg"
    plt.savefig(str(final_path), dpi=130, bbox_inches='tight')
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Image            : {args.image}")
    print(f"Classes used     : {', '.join(selected_classes)}")
    print(f"Output directory : {out_dir}")
    print(f"Total iterations : {n_iters}")
    print(f"Initial 1-SSIM   : {ssim_history[0]:.5f}")
    print(f"Final   1-SSIM   : {ssim_history[-1]:.5f}")
    print(f"Total improvement: {ssim_history[0] - ssim_history[-1]:.5f}")
    print(f"Convergence curve: {conv_path}")
    print(f"Final comparison : {final_path}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
