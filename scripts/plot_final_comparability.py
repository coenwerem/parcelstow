"""Render the audited final evaluation export without running simulation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

_CMU_BOLD = Path("/usr/share/fonts/truetype/cmu/cmunsx.ttf")
_CMU_REG  = Path("/usr/share/fonts/truetype/cmu/cmunss.ttf")
if _CMU_BOLD.exists():
    fm.fontManager.addfont(str(_CMU_BOLD))
if _CMU_REG.exists():
    fm.fontManager.addfont(str(_CMU_REG))

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["CMU Sans Serif", "DejaVu Sans", "Arial"],
    "font.size":          13,
    "axes.labelsize":     18,
    "axes.titlesize":     20,
    "axes.titleweight":   "bold",
    "axes.labelweight":   "normal",
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.linewidth":     1,
    "legend.fontsize":    12,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "figure.dpi":         600,
    "grid.alpha":         0.12,
    "grid.linewidth":     0.3,
    "text.usetex":        False,
    "axes.unicode_minus": False,
})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.evidence.read_text())['results']
    args.output.mkdir(parents=True)
    for kind in ('success', 'difference'):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 3, figure=fig, left=.08, right=.99,
                               bottom=.12, top=.87, wspace=.24, hspace=.45)
        for ti, task in enumerate(('upright', 'peg')):
            for seed in range(3):
                ax = fig.add_subplot(gs[ti, seed])
                rows = sorted([r for r in data if r['task'] == task and r['seed'] == seed],
                              key=lambda r: r['rate'])
                x = [r['rate'] for r in rows]
                ax.axvspan(rows[0]['demonstrated_lo'], rows[0]['demonstrated_hi'],
                           color='#9e9e9e', alpha=.16)
                if kind == 'success':
                    ax.plot(x, [r['expert_successes'] / 2 for r in rows], 'o-',
                            color='#0d47a1', label='Expert')
                    ax.plot(x, [r['act_successes'] / 2 for r in rows], 's--',
                            color='#bf360c', label='ACT')
                    ax.set_ylim(-3, 103)
                else:
                    y = [r['difference_pp'] for r in rows]
                    ax.errorbar(x, y, yerr=[[v-r['ci_lower_pp'] for v, r in zip(y, rows)],
                                          [r['ci_upper_pp']-v for v, r in zip(y, rows)]],
                                color='#bf360c', fmt='o-', capsize=3)
                    ax.axhline(0, color='#9e9e9e', linestyle=':')
                    ax.set_ylim(-105, 30)
                ax.set_title(f'{task.capitalize()} · seed {seed}' + (' (primary)' if seed == 0 else ''),
                             fontsize=17)
                ax.set_xticks([.5, 1, 1.5, 2, 2.5])
                ax.grid(True)
                if ti == 1:
                    ax.set_xlabel('Execution rate')
                if seed == 0:
                    ax.set_ylabel('Task success (%)' if kind == 'success' else 'ACT − expert (pp)')
                if kind == 'success' and ti == seed == 0:
                    ax.legend(loc='lower left')
        title = ('Final task success: 200 episodes per actor and rate' if kind == 'success'
                 else 'Paired differences: pointwise 95% bootstrap intervals')
        fig.suptitle(title, y=.97, fontsize=21)
        fig.text(.5, .025, 'Gray shading: demonstrated rate range. Seeds are shown separately; expert trials are not pooled.',
                 ha='center', fontsize=13)
        for suffix in ('png', 'pdf'):
            fig.savefig(args.output / f'{kind}.{suffix}', dpi=600, bbox_inches='tight',
                        facecolor='white', pad_inches=.02)
        fig.savefig(args.output / f'{kind}-preview.png', dpi=110, facecolor='white')
        plt.close(fig)


if __name__ == '__main__':
    main()
