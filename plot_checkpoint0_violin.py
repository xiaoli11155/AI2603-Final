import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def load_winrates(csv_path):
    baseline = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            round_label = row.get('Round', '').strip().lower()
            if round_label == 'average':
                continue
            try:
                baseline.append(float(row['vs RL Baseline Winrate']))
            except (KeyError, ValueError):
                continue
    if not baseline:
        raise ValueError('No winrate rows found in CSV')
    return baseline


def plot_violin(winrates, out_path):
    df = pd.DataFrame({'Winrate': winrates, 'x': [''] * len(winrates)})

    # ====== 全局风格 ======
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    plt.figure(figsize=(6, 4))

    # ====== 左半边：小提琴图 ======
    ax = sns.violinplot(
        data=df,
        x="x",
        y="Winrate",
        inner=None,
        cut=2,
        bw_adjust=1,
        linewidth=0,
        color="#FFC5AE",
        width=0.8
    )

    # 裁成左半边
    for c in ax.collections:
        x0, y0, w, h = c.get_paths()[0].get_extents().bounds
        c.set_clip_path(
            plt.Rectangle((x0, y0), w / 2, h, transform=ax.transData)
        )

    # ====== 左半边：箱线图 ======
    sns.boxplot(
        data=df,
        x="x",
        y="Winrate",
        width=0.18,
        showcaps=True,
        showfliers=False,
        boxprops={
            "facecolor": "#EE8C79",
            "edgecolor": "#CD482B",
            "alpha": 0.85,
            "zorder": 3,
            "linewidth": 2
        },
        whiskerprops={"color": "#CD482B", "linewidth": 2},
        capprops={"color": "#CD482B", "linewidth": 2},
        medianprops={"color": "black", "linewidth": 2},
        ax=ax
    )

    # 将箱线也移到左侧
    for artist in ax.artists:
        artist.set_x(artist.get_x() - 0.1)

    # ====== 右半边：点云（strip / swarm） ======
    # ====== 右半边：更密集 + 更大的点云 ======
    sns.stripplot(
        data=df,
        x="x",
        y="Winrate",
        jitter=0.18,     # 更小抖动 → 更集中
        size=7,         # 点更大
        alpha=1,
        color="#E36F49",
        ax=ax
    )

    # 将点整体向右平移（比之前更远）
    point_shift = 0.28
    for coll in ax.collections:
        if hasattr(coll, "get_offsets"):
            offsets = coll.get_offsets()
            offsets[:, 0] += point_shift
            coll.set_offsets(offsets)


    # ====== 坐标轴美化 ======
    ax.set_title("Win Rate Distribution over 100 Evaluation Rounds", fontsize=14, pad=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_xlabel("")

    ax.set_ylim(0.4, 0.8)
    ax.set_yticks(np.linspace(0.4, 0.8, 5))

    ax.tick_params(axis='both', labelsize=11)
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format='pdf')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot violin charts for checkpoint0 evaluation winrates.')
    parser.add_argument('--input', type=str, default='checkpoint0_eval_results.csv', help='CSV file with winrate columns')
    parser.add_argument('--output', type=str, default='checkpoint0_violin.pdf', help='Output PDF path')
    args = parser.parse_args()

    winrates = load_winrates(args.input)
    plot_violin(winrates, args.output)
    print(f'Saved violin plot to {args.output}')


if __name__ == '__main__':
    main()
