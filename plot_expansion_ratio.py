import json
from pathlib import Path
import matplotlib.font_manager as fm
import os
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as error:
    print(
        "Missing plotting dependency. Please install matplotlib before running this script.\n"
        f"Details: {error}"
    )
    raise SystemExit(1) from error


font_path = 'SourceHanSansSC-Regular.otf'

# 注册字体
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    # 获取字体名称
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    print(f"注册字体: {font_name}")
else:
    print(f"字体文件不存在: {font_path}")
    font_name = 'DejaVu Sans'  # 回退

# 设置全局字体
plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_DATA_FILE = "expansion_ratio_data.json"
DEFAULT_OUTPUT_FILE = "expansion_ratio.png"


def load_data(json_file):
    """Load benchmark LOC data from a JSON file."""
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)


def collect_valid_benchmarks(data):
    """Keep only benchmarks whose DSL and TIR LOC are both available."""
    valid_benchmarks = []
    for benchmark, values in data.items():
        dsl_loc = values.get("dsl_loc")
        tir_loc = values.get("tir_loc")
        if dsl_loc is not None and tir_loc is not None:
            valid_benchmarks.append(benchmark)
    return valid_benchmarks


def annotate_bars(ax, bars, values, formatter, y_multiplier=1.08):
    """Render numeric labels above bars on a log-scale y-axis."""
    for bar, value in zip(bars, values):
        if value is None or value <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * y_multiplier,
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )


def plot_expansion_ratio(data, output_file=DEFAULT_OUTPUT_FILE):
    """Plot DSL LOC, lowered TIR LOC, and CER for each benchmark."""
    valid_benchmarks = collect_valid_benchmarks(data)
    if not valid_benchmarks:
        print(
            "No valid data to plot yet. Please fill in both 'dsl_loc' and 'tir_loc' "
            f"in {DEFAULT_DATA_FILE}."
        )
        return

    dsl_values = [data[benchmark]["dsl_loc"] for benchmark in valid_benchmarks]
    tir_values = [data[benchmark]["tir_loc"] for benchmark in valid_benchmarks]
    cer_values = [tir / dsl for dsl, tir in zip(dsl_values, tir_values)]

    x = np.arange(len(valid_benchmarks))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_dsl = ax.bar(
        x - width / 2,
        dsl_values,
        width,
        label="领域特定语言",
        color="#3498db",
        edgecolor="black",
        alpha=0.9,
    )
    bars_tir = ax.bar(
        x + width / 2,
        tir_values,
        width,
        label="中间表示",
        color="#f39c12",
        edgecolor="black",
        alpha=0.9,
    )

    ax.set_yscale("log")
    ax.set_ylabel("代码行数", fontsize=16, fontweight="bold")
    # ax.set_title(
    #     "编译降级前后的代码行数对比与抽象膨胀率",
    #     fontsize=14,
    #     fontweight="bold",
    #     pad=18,
    # )
    ax.set_xticks(x)
    ax.set_xticklabels(valid_benchmarks, fontsize=14, fontweight="bold")
    ax.legend(fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    ymax = max(tir_values)
    ax.set_ylim(bottom=max(min(dsl_values), 1) * 0.8, top=ymax * 2.0)

    annotate_bars(
        ax,
        bars_dsl,
        dsl_values,
        formatter=lambda value: f"{int(value)}",
        y_multiplier=1.04,
    )
    annotate_bars(
        ax,
        bars_tir,
        tir_values,
        formatter=lambda value: f"{int(value)}",
        y_multiplier=1.06,
    )

    for i, cer in enumerate(cer_values):
        ratio_y_multiplier = 1.28
        ax.text(
            x[i] + width / 2,
            tir_values[i] * ratio_y_multiplier,
            f"{cer:.1f}x",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="#d35400",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Chart successfully saved to {output_file}")


if __name__ == "__main__":
    data_path = Path(DEFAULT_DATA_FILE)
    if not data_path.exists():
        print(
            f"Data file '{DEFAULT_DATA_FILE}' was not found. "
            "Please create it or keep it next to this script."
        )
    else:
        benchmark_data = load_data(data_path)
        plot_expansion_ratio(benchmark_data)
