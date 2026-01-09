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


def plot_violin(df, out_path):
    # ====== 全局风格 ======
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    plt.figure(figsize=(6, 6))

    # 定义颜色映射
    # V1: 绿色系 (左侧)
    # V2: 橙色系 (右侧)
    # 注意: sns.violinplot 默认按照 x 分类的顺序排列
    violin_colors = {"V1": "#C8E6C9", "V2": "#FFC680"}
    box_colors = {"V1": "#81C784", "#E5926C": "#E5926C"} # V2 uses original
    box_colors = {"V1": "#81C784", "V2": "#E5926C"}
    edge_colors = {"V1": "#2E7D32", "V2": "#B14D25"}

    # ====== 小提琴图 ======
    ax = sns.violinplot(
        data=df,
        x="Group",
        y="Winrate",
        inner=None,          # 不画内部结构，后面叠加箱线
        cut=2,               # 不外推
        bw_adjust=1,         # 保持原样
        linewidth=0,
        palette=violin_colors,
        order=["V1", "V2"] # 确保 V1 在左边
    )

    # ====== 叠加箱线图 ======
    # 为每个组分别叠加箱线图
    groups = ["V1", "V2"]
    for i, group in enumerate(groups):
        group_data = df[df["Group"] == group]
        if group_data.empty: continue
        
        # 计算位置
        pos = i 
        
        sns.boxplot(
            y=group_data["Winrate"],
            x=[group] * len(group_data),
            width=0.18,
            showcaps=True,
            showfliers=False,
            boxprops={
                "facecolor": box_colors[group],
                "edgecolor": edge_colors[group],
                "alpha": 0.85,
                "zorder": 3,
                "linewidth": 2
            },
            whiskerprops={
                "color": edge_colors[group],
                "linewidth": 2
            },
            capprops={
                "color": edge_colors[group],
                "linewidth": 2
            },
            medianprops={
                "color": "black",
                "linewidth": 2.5
            },
            ax=ax
        )

    # ====== 坐标轴美化 ======
    ax.set_title("Win Rate Distribution: V1 vs V2", fontsize=14, pad=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_xlabel("Evaluation Function", fontsize=12)

    ax.set_ylim(0.3, 1.0)
    ax.set_yticks(np.linspace(0.3, 1.0, 8))

    ax.tick_params(axis='both', labelsize=11)

    # 去掉右上边框
    # sns.despine(top=True, right=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format='pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot violin charts for minimax evaluation winrates.')
    parser.add_argument('--input1', type=str, default='minimax_eval_results_norm1.csv', help='CSV for V1 (Green)')
    parser.add_argument('--input2', type=str, default='minimax_eval_results_norm2.csv', help='CSV for V2 (Orange)')
    parser.add_argument('--output', type=str, default='minimax_violin_comparison.pdf', help='Output PDF path')
    args = parser.parse_args()

    winrates1 = load_winrates(args.input1)
    winrates2 = load_winrates(args.input2)

    df1 = pd.DataFrame({'Winrate': winrates1, 'Group': 'V1'})
    df2 = pd.DataFrame({'Winrate': winrates2, 'Group': 'V2'})
    df = pd.concat([df1, df2], ignore_index=True)

    plot_violin(df, args.output)
    print(f'Saved comparison violin plot to {args.output}')


if __name__ == '__main__':
    main()
