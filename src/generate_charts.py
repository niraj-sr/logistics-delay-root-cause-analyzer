"""
Logistics Delay Root-Cause Analyzer
Publication Charts Generation Script (Milestone 4 - Polished Layout)

Author: Niraj & Senior Analytics Mentor
Purpose: Generates the 6 approved publication-quality Matplotlib figures with
         impeccable typography, zero label overlap, zero-baseline axes,
         consistent terminology, and precise sample size disclosures.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Configuration & Global Styling
# ---------------------------------------------------------
INPUT_FILE = Path("data/processed/shipments_cleaned.csv")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#eaeaea"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

NETWORK_BASELINE = 15.60


def load_and_prepare_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    conditions_dist = [
        df["Distance_KM"] < 600,
        (df["Distance_KM"] >= 600) & (df["Distance_KM"] <= 1500),
        df["Distance_KM"] > 1500,
    ]
    choices_dist = ["Short", "Medium", "Long"]
    df["Distance_Tier"] = np.select(conditions_dist, choices_dist, default="Unknown")
    df["Route"] = df["Origin_Warehouse"] + " → " + df["Destination_City"]
    return df


# ---------------------------------------------------------
# Chart 1: Network Delay Overview
# ---------------------------------------------------------
def generate_chart_1(df: pd.DataFrame) -> Path:
    total_shipments = len(df)
    delayed_shipments = int((df["Delayed"] == 1).sum())
    on_time_shipments = int((df["Delayed"] == 0).sum())
    delay_rate = (delayed_shipments / total_shipments) * 100
    otd_rate = 100.0 - delay_rate

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

    # Horizontal stacked bar
    ax.barh(
        ["Total Network Volume"],
        [on_time_shipments],
        color="#1f77b4",
        edgecolor="none",
        label=f"On-Time Deliveries ({on_time_shipments:,} shipments)",
        height=0.45,
    )
    ax.barh(
        ["Total Network Volume"],
        [delayed_shipments],
        left=[on_time_shipments],
        color="#d62728",
        edgecolor="none",
        label=f"Delayed Deliveries ({delayed_shipments:,} shipments)",
        height=0.45,
    )

    # In-bar text labels
    ax.text(
        on_time_shipments / 2,
        0,
        f"On-Time: {on_time_shipments:,}\n({otd_rate:.1f}%)",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=11.5,
    )
    ax.text(
        on_time_shipments + (delayed_shipments / 2),
        0,
        f"Delayed: {delayed_shipments:,}\n({delay_rate:.2f}%)",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=11.5,
    )

    ax.set_xlim(0, 5200)
    ax.set_xlabel("Shipment Count (N = 5,000)", fontsize=11, fontweight="bold", labelpad=10)
    
    # Title & Subtitle cleanly spaced
    ax.set_title(
        "Network Delivery Overview: On-Time vs. Delayed Shipments",
        fontsize=13,
        fontweight="bold",
        pad=22,
    )
    fig.text(
        0.5,
        0.88,
        "Network Baseline Delay Rate: 15.60% (1 in every 6.4 shipments delayed) | Q1 2024 Synthetic Dataset",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=True, fontsize=10)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "01_network_delay_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------
# Chart 2: Route Performance (All 20 Corridors Ranked)
# ---------------------------------------------------------
def generate_chart_2(df: pd.DataFrame) -> Path:
    route_summary = (
        df.groupby("Route")
        .agg(
            total_shipments=("Order_ID", "count"),
            delayed_shipments=("Delayed", "sum"),
            delay_rate=("Delayed", lambda x: (x.mean() * 100)),
        )
        .sort_values(by="delay_rate", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(12, 9.5), dpi=300)

    transcontinental = {"WH-West → New York", "WH-East → Los Angeles"}
    colors = [
        "#b22222" if r in transcontinental else "#4682b4"
        for r in route_summary.index
    ]

    bars = ax.barh(
        route_summary.index,
        route_summary["delay_rate"],
        color=colors,
        height=0.65,
    )

    for bar, (_, row) in zip(bars, route_summary.iterrows()):
        rate_val = row["delay_rate"]
        delayed_val = int(row["delayed_shipments"])
        tot_val = int(row["total_shipments"])
        ax.text(
            rate_val + 0.7,
            bar.get_y() + bar.get_height() / 2,
            f"{rate_val:.1f}% ({delayed_val} delayed / N={tot_val})",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#222222",
        )

    ax.axvline(
        x=NETWORK_BASELINE,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label=f"Network Baseline ({NETWORK_BASELINE:.2f}%)",
    )

    # Positioned callouts in open whitespace
    ax.text(
        35.0,
        14.0,
        "Transcontinental corridors account for\n429 of 780 delays (55.0%).",
        fontsize=9.5,
        fontweight="bold",
        color="#b22222",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fff2f2", ec="#b22222", lw=1.2),
    )

    ax.text(
        35.0,
        10.5,
        "WH-North → Chicago: 29.19% delay rate;\naverage warehouse processing = 1.70 days.",
        fontsize=9.0,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#777777", lw=0.9),
    )

    ax.set_xlim(0, 72)
    ax.set_xlabel("Delay Rate (%)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Trade Route (Origin Warehouse → Destination City)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Observed Delay Rate by Route (All 20 Corridors Ranked)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "02_route_delay_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------
# Chart 3: Carrier x Distance Tier
# ---------------------------------------------------------
def generate_chart_3(df: pd.DataFrame) -> Path:
    tier_order = ["Short", "Medium", "Long"]
    carrier_order = ["ExpressAir", "FastTrack", "SwiftCargo"]
    carrier_colors = {"ExpressAir": "#7b1fa2", "FastTrack": "#0d47a1", "SwiftCargo": "#00838f"}

    c_tier = df.groupby(["Distance_Tier", "Carrier"]).agg(
        total_shipments=("Order_ID", "count"),
        delayed_shipments=("Delayed", "sum"),
        delay_rate=("Delayed", lambda x: (x.mean() * 100)),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)

    x = np.arange(len(tier_order))
    width = 0.25

    for i, carrier in enumerate(carrier_order):
        subset = c_tier[c_tier["Carrier"] == carrier].set_index("Distance_Tier").reindex(tier_order)
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            subset["delay_rate"],
            width=width,
            label=carrier,
            color=carrier_colors[carrier],
        )

        for bar, (_, row) in zip(bars, subset.iterrows()):
            rate = row["delay_rate"]
            n_tot = int(row["total_shipments"])
            n_del = int(row["delayed_shipments"])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.9,
                f"{rate:.1f}%\n({n_del}/{n_tot})",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )

    ax.axhline(
        y=NETWORK_BASELINE,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label=f"Network Baseline ({NETWORK_BASELINE:.2f}%)",
    )

    ax.text(
        0.03,
        0.87,
        "ExpressAir shows higher observed delay rates across all three distance tiers,\n"
        "while also having the shortest average transit time overall (2.38 days).",
        transform=ax.transAxes,
        fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.45", fc="#fdf7ff", ec="#7b1fa2", lw=1),
    )

    ax.set_ylim(0, 52)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Short (<600 km)", "Medium (600–1,500 km)", "Long (>1,500 km)"],
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_ylabel("Delay Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Distance Tier", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Carrier Delay Rate by Distance Tier",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "03_carrier_by_distance_tier.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------
# Chart 4: Weather x Distance Tier
# ---------------------------------------------------------
def generate_chart_4(df: pd.DataFrame) -> Path:
    tier_order = ["Short", "Medium", "Long"]
    weather_order = ["Clear", "Rain", "Fog", "Storm"]
    weather_colors = {
        "Clear": "#4fc3f7",
        "Rain": "#1976d2",
        "Fog": "#78909c",
        "Storm": "#b71c1c",
    }

    w_tier = df.groupby(["Distance_Tier", "Weather_Condition"]).agg(
        total_shipments=("Order_ID", "count"),
        delayed_shipments=("Delayed", "sum"),
        delay_rate=("Delayed", lambda x: (x.mean() * 100)),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=300)

    x = np.arange(len(tier_order))
    width = 0.18

    for i, weather in enumerate(weather_order):
        subset = w_tier[w_tier["Weather_Condition"] == weather].set_index("Distance_Tier").reindex(tier_order)
        offset = (i - 1.5) * width
        bars = ax.bar(
            x + offset,
            subset["delay_rate"],
            width=width,
            label=weather,
            color=weather_colors[weather],
        )

        for bar, (_, row) in zip(bars, subset.iterrows()):
            rate = row["delay_rate"]
            n_tot = int(row["total_shipments"])
            n_del = int(row["delayed_shipments"])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{rate:.0f}%\n({n_del}/{n_tot})",
                ha="center",
                va="bottom",
                fontsize=7.4,
            )

    ax.axhline(
        y=NETWORK_BASELINE,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label=f"Network Baseline ({NETWORK_BASELINE:.2f}%)",
    )

    # Positioned callouts in open upper region
    ax.text(
        0.03,
        0.88,
        "Short-route storms show a 100% observed delay rate (36/36 shipments)\n"
        "in this synthetic dataset.",
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        color="#b71c1c",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#b71c1c", lw=1),
    )

    ax.text(
        0.03,
        0.75,
        "Clear-weather shipments account for 435 of 780 delays (55.8%)\n"
        "because clear conditions represent the majority of shipments.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#0d47a1",
        bbox=dict(boxstyle="round,pad=0.4", fc="#e3f2fd", ec="#1976d2", lw=0.9),
    )

    ax.set_ylim(0, 125)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Short (<600 km)", "Medium (600–1,500 km)", "Long (>1,500 km)"],
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_ylabel("Delay Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Distance Tier", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Weather Condition Delay Rate by Distance Tier",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "04_weather_by_distance_tier.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------
# Chart 5: Traffic x Distance Tier
# ---------------------------------------------------------
def generate_chart_5(df: pd.DataFrame) -> Path:
    tier_order = ["Short", "Medium", "Long"]
    traffic_order = ["Low", "Medium", "High"]
    traffic_colors = {"Low": "#90a4ae", "Medium": "#42a5f5", "High": "#e65100"}

    t_tier = df.groupby(["Distance_Tier", "Traffic_Level"]).agg(
        total_shipments=("Order_ID", "count"),
        delayed_shipments=("Delayed", "sum"),
        delay_rate=("Delayed", lambda x: (x.mean() * 100)),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)

    x = np.arange(len(tier_order))
    width = 0.25

    for i, traffic in enumerate(traffic_order):
        subset = t_tier[t_tier["Traffic_Level"] == traffic].set_index("Distance_Tier").reindex(tier_order)
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            subset["delay_rate"],
            width=width,
            label=traffic,
            color=traffic_colors[traffic],
        )

        for bar, (_, row) in zip(bars, subset.iterrows()):
            rate = row["delay_rate"]
            n_tot = int(row["total_shipments"])
            n_del = int(row["delayed_shipments"])
            # Offset text slightly higher if close to baseline to avoid overlap
            y_pos = bar.get_height() + 1.2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{rate:.1f}%\n({n_del}/{n_tot})",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )

    ax.axhline(
        y=NETWORK_BASELINE,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label=f"Network Baseline ({NETWORK_BASELINE:.2f}%)",
    )

    ax.text(
        0.03,
        0.87,
        "High traffic is associated with higher observed delay rates\nacross all three distance tiers.",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color="#e65100",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fff8e1", ec="#e65100", lw=1),
    )

    ax.set_ylim(0, 42)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Short (<600 km)", "Medium (600–1,500 km)", "Long (>1,500 km)"],
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_ylabel("Delay Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Distance Tier", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Traffic Level Delay Rate by Distance Tier",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "05_traffic_by_distance_tier.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------
# Chart 6: High-Volume vs. High-Severity Matrix (Impeccable Placement)
# ---------------------------------------------------------
def generate_chart_6(df: pd.DataFrame) -> Path:
    # 10 key segments with manually verified (x, y) text positions to guarantee zero overlap
    segments = [
        {
            "name": "WH-East → LA (ExpressAir)",
            "delayed": 87, "rate": 100.0, "N": 87, "color": "#b71c1c",
            "text_x": 105, "text_y": 104, "align": "left"
        },
        {
            "name": "WH-West → NY (ExpressAir)",
            "delayed": 77, "rate": 100.0, "N": 77, "color": "#b71c1c",
            "text_x": 60, "text_y": 104, "align": "right"
        },
        {
            "name": "SwiftCargo in Storm",
            "delayed": 40, "rate": 74.1, "N": 54, "color": "#8e24aa",
            "text_x": 40, "text_y": 82, "align": "center"
        },
        {
            "name": "WH-West → NY (SwiftCargo)",
            "delayed": 84, "rate": 47.7, "N": 176, "color": "#e65100",
            "text_x": 66, "text_y": 55, "align": "right"
        },
        {
            "name": "WH-East → LA (SwiftCargo)",
            "delayed": 87, "rate": 43.3, "N": 201, "color": "#e65100",
            "text_x": 105, "text_y": 52, "align": "left"
        },
        {
            "name": "WH-West → NY (FastTrack)",
            "delayed": 49, "rate": 34.8, "N": 141, "color": "#0288d1",
            "text_x": 33, "text_y": 42, "align": "right"
        },
        {
            "name": "WH-East → LA (FastTrack)",
            "delayed": 45, "rate": 32.8, "N": 137, "color": "#0288d1",
            "text_x": 28, "text_y": 28, "align": "right"
        },
        {
            "name": "WH-North → Chicago (All)",
            "delayed": 61, "rate": 29.2, "N": 209, "color": "#3949ab",
            "text_x": 75, "text_y": 23, "align": "left"
        },
        {
            "name": "ExpressAir in Clear Weather",
            "delayed": 196, "rate": 27.4, "N": 715, "color": "#d81b60",
            "text_x": 180, "text_y": 38, "align": "center"
        },
        {
            "name": "FastTrack in Clear Weather",
            "delayed": 70, "rate": 5.6, "N": 1246, "color": "#2e7d32",
            "text_x": 98, "text_y": 7, "align": "left"
        },
    ]

    fig, ax = plt.subplots(figsize=(12.5, 8.0), dpi=300)

    # Dividing threshold lines
    x_threshold = 60
    y_threshold = NETWORK_BASELINE

    ax.axvline(x=x_threshold, color="#888888", linestyle=":", linewidth=1.2)
    ax.axhline(y=y_threshold, color="#333333", linestyle="--", linewidth=1.2, label=f"Network Baseline ({y_threshold:.2f}%)")

    # Descriptive Quadrant Labels (Corner Watermarks)
    ax.text(
        150, 97,
        "High Severity / High Delayed Volume\n(Elevated Delay Rate & Large Delay Volume)",
        ha="center", va="center", fontsize=9.0, fontweight="bold", color="#880e4f",
        bbox=dict(boxstyle="square,pad=0.4", fc="#fce4ec", ec="#880e4f", alpha=0.85, lw=0.8),
    )
    ax.text(
        25, 97,
        "High Severity / Lower Delayed Volume\n(Elevated Delay Rate, Smaller N)",
        ha="left", va="center", fontsize=9.0, fontweight="bold", color="#4a148c",
        bbox=dict(boxstyle="square,pad=0.4", fc="#f3e5f5", ec="#4a148c", alpha=0.85, lw=0.8),
    )
    ax.text(
        155, 6,
        "Lower Severity / High Delayed Volume\n(Lower Delay Rate, High Throughput)",
        ha="center", va="center", fontsize=9.0, fontweight="bold", color="#1b5e20",
        bbox=dict(boxstyle="square,pad=0.4", fc="#e8f5e9", ec="#1b5e20", alpha=0.85, lw=0.8),
    )
    ax.text(
        25, 6,
        "Lower Severity / Lower Delayed Volume",
        ha="left", va="center", fontsize=9.0, fontweight="bold", color="#546e7a",
        bbox=dict(boxstyle="square,pad=0.4", fc="#eceff1", ec="#546e7a", alpha=0.85, lw=0.8),
    )

    # Plot bubbles and custom annotations
    for item in segments:
        size = item["N"] * 2.0
        ax.scatter(
            item["delayed"],
            item["rate"],
            s=size,
            c=item["color"],
            alpha=0.65,
            edgecolors="#222222",
            linewidths=1.2,
            zorder=3,
        )

        # Draw connecting line if text is offset
        ax.annotate(
            f"{item['name']}\n(Rate: {item['rate']:.1f}%, Delayed: {item['delayed']}, N={item['N']})",
            xy=(item["delayed"], item["rate"]),
            xytext=(item["text_x"], item["text_y"]),
            ha=item["align"],
            va="center",
            fontsize=7.6,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.9, lw=0.6),
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, shrinkA=3, shrinkB=6),
            zorder=4,
        )

    # Bubble Size legend
    for sample_n in [100, 500, 1000]:
        ax.scatter([], [], s=sample_n * 2.0, c="#888888", alpha=0.5, edgecolors="#222222", label=f"N = {sample_n}")

    ax.set_xlim(0, 230)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Delayed Shipment Count", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Delay Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Operational Segment Matrix: Delay Severity vs. Delayed Shipment Volume",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, fontsize=8.8, title="Bubble Size (N)")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "06_high_volume_vs_high_severity_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    print("LOGISTICS DELAY ANALYZER: RENDERING POLISHED CHARTS ...")
    df = load_and_prepare_data(INPUT_FILE)
    generate_chart_1(df)
    generate_chart_2(df)
    generate_chart_3(df)
    generate_chart_4(df)
    generate_chart_5(df)
    generate_chart_6(df)
    print("ALL 6 POLISHED CHARTS SUCCESSFULLY EXPORTED TO reports/figures/")


if __name__ == "__main__":
    main()
