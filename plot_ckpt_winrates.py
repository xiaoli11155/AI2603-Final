import argparse
import json
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_csv(records, out_csv):
    if not records:
        return
    keys = ['checkpoint', 'winrate', 'wins', 'games']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in keys})


def plot(records, out_png):
    if not records:
        raise ValueError('No records to plot')

    # Use file order as x (preserves test order)
    x = list(range(len(records)))
    y = [r.get('winrate', 0.0) for r in records]
    labels = [os.path.basename(r.get('checkpoint', '')) for r in records]

    plt.figure(figsize=(10, 7))
    plt.plot(x, y,  linestyle='-')
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.4)
    # plt.xlabel('Checkpoint')
    plt.ylabel('Winrate', fontsize=16)
    # plt.title('Checkpoint Winrates')

    # annotate a few x labels to avoid clutter: first, last and every 10th
    xticks = x[::max(1, len(x)//10)] if len(x) > 20 else x
    if 0 not in xticks:
        xticks = [0] + xticks
    if x[-1] not in xticks:
        xticks.append(x[-1])
    plt.xticks(xticks, [labels[i] for i in xticks], rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='ckpt_winrates.json')
    parser.add_argument('--output', type=str, default='ckpt_winrates.png')
    parser.add_argument('--csv', type=str, default=None, help='Optional CSV output')
    args = parser.parse_args()

    records = load_results(args.input)
    plot(records, args.output)
    if args.csv:
        save_csv(records, args.csv)
    print(f'Saved plot to {args.output}')


if __name__ == '__main__':
    main()
