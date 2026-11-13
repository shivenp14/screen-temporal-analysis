"""Describe source labels over time; these figures are not causal estimates."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "screen-temporal-v1"
import matplotlib.pyplot as plt

LABELS = ("Negative", "Neutral", "Positive")
COLORS = ("#bd4a43", "#7b8794", "#287e70")


def save_svg(fig, path):
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")


def summarize(rows, frequency):
    counts = Counter()
    for row in rows:
        date = row.get("publication_date")
        if not date:
            continue
        period = date[:7] if frequency == "month" else f"{date[:4]}-Q{(int(date[5:7])-1)//3+1}"
        counts[(row["platform"], period, row["sentiment_3class"])] += 1
    result = []
    for platform, period in sorted({key[:2] for key in counts}):
        n = sum(counts[(platform, period, label)] for label in LABELS)
        for label in LABELS:
            count = counts[(platform, period, label)]
            result.append({"platform": platform, "period": period, "label": label,
                           "count": count, "n": n, "proportion": count/n, "small_bin": n < 10})
    return result


def run(dataset, output):
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    output.mkdir(parents=True, exist_ok=True)
    for frequency in ("month", "quarter"):
        table = summarize(rows, frequency)
        if not table:
            raise ValueError("No dated records for temporal analysis")
        with (output / f"{frequency}_sentiment.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        all_periods = sorted({r["period"] for r in table})
        for ax, platform in zip(axes, ("youtube", "snapchat")):
            lookup = {(r["period"], r["label"]): r for r in table if r["platform"] == platform}
            for label, color in zip(LABELS, COLORS):
                values = [lookup[(p, label)]["proportion"] if (p, label) in lookup else float("nan") for p in all_periods]
                ax.plot(all_periods, values, marker="o", markersize=4, label=label, color=color)
            ax.set_title(platform.title()); ax.set_ylim(-0.03, 1.03); ax.set_ylabel("Share of source labels")
            ax.tick_params(axis="x", rotation=45); ax.grid(alpha=.2); ax.legend(loc="upper left", ncol=3)
        interval = "Monthly" if frequency == "month" else "Quarterly"
        fig.suptitle(f"Accepted intermediate SCREEN records: {interval} sentiment\nDeduplicated within platform; Mixed mapped to Neutral", fontsize=12)
        save_svg(fig, output / f"{frequency}_sentiment.svg")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, platform in zip(axes, ("youtube", "snapchat")):
        values = [sum(r["platform"] == platform and r["sentiment_3class"] == label for r in rows) for label in LABELS]
        bars = ax.bar(LABELS, values, color=COLORS)
        ax.bar_label(bars); ax.set_title(f"{platform.title()} (n={sum(values)})"); ax.set_ylabel("Article/platform records")
    fig.suptitle("Accepted intermediate labels after within-platform deduplication")
    save_svg(fig, output / "platform_labels.svg"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/screen/articles.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/temporal"))
    args = parser.parse_args()
    run(args.dataset, args.output)


if __name__ == "__main__":
    main()
