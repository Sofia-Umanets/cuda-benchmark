"""
Модуль визуализации результатов бенчмарков.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def _overhead_color(value):
    """Цвет в зависимости от величины overhead."""
    if abs(value) < 2:
        return "#2ecc71"  # зелёный — нет разницы
    if abs(value) < 5:
        return "#f39c12"  # оранжевый — небольшая
    return "#e74c3c"      # красный — заметная


def plot_by_category(df, output):
    """Overhead по категориям (столбчатая диаграмма)."""
    by_cat = df.groupby("category")["overhead_%"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(by_cat))
    colors = [_overhead_color(m) for m in by_cat["mean"]]

    ax.bar(x, by_cat["mean"], yerr=by_cat["std"], capsize=4,
           color=colors, edgecolor="black", linewidth=0.8, alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(5, color="red", linewidth=1, linestyle="--", alpha=0.4)
    ax.axhline(-5, color="red", linewidth=1, linestyle="--", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(by_cat["category"], rotation=45, ha="right")
    ax.set_ylabel("Docker Overhead (%)")
    ax.set_title("Overhead по категориям")

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distribution(df, output):
    """Распределение overhead (гистограмма + boxplot)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Гистограмма
    ax1.hist(df["overhead_%"], bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(df["overhead_%"].mean(), color="red", linestyle="--",
                label=f"среднее: {df['overhead_%'].mean():.2f}%")
    ax1.axvline(0, color="black", linewidth=1)
    ax1.set_xlabel("Overhead (%)")
    ax1.set_ylabel("Количество")
    ax1.set_title("Распределение overhead")
    ax1.legend()

    # Boxplot
    categories = df["category"].unique()
    data = [df[df["category"] == cat]["overhead_%"].values for cat in categories]
    bp = ax2.boxplot(data, labels=categories, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("Overhead (%)")
    ax2.set_title("Overhead по категориям")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scatter(df, output):
    """Scatter: native vs docker."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ["red" if s else "steelblue" for s in df["significant"]]
    ax.scatter(df["native_ms"], df["docker_ms"], c=colors, alpha=0.6, s=50)

    lim = max(df["native_ms"].max(), df["docker_ms"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y = x")
    ax.plot([0, lim], [0, lim * 1.05], "r--", alpha=0.3)
    ax.plot([0, lim], [0, lim * 0.95], "r--", alpha=0.3, label="±5%")

    ax.set_xlabel("Native (ms)")
    ax.set_ylabel("Docker (ms)")
    ax.set_title("Native vs Docker")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def plot_detailed(df, output):
    """Детальный overhead по всем тестам."""
    df_sorted = df.sort_values("overhead_%")

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.25)))
    colors = [_overhead_color(v) for v in df_sorted["overhead_%"]]
    y = np.arange(len(df_sorted))

    ax.barh(y, df_sorted["overhead_%"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["name"], fontsize=8)
    ax.set_xlabel("Overhead (%)")
    ax.set_title("Overhead по тестам")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(5, color="red", linewidth=1, linestyle="--", alpha=0.4)
    ax.axvline(-5, color="red", linewidth=1, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def create_all_plots(df, output_dir):
    """
    Создать все графики.
    
    Returns:
        list: пути к созданным файлам
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ("overhead_by_category.png", plot_by_category),
        ("distribution.png", plot_distribution),
        ("scatter.png", plot_scatter),
        ("detailed.png", plot_detailed),
    ]

    created = []
    for filename, plot_func in plots:
        path = figures_dir / filename
        plot_func(df, path)
        created.append(path)

    return created