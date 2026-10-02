"""
Logistics Delay Root-Cause Analyzer
Dataset Generator (Synthetic US Logistics Scenario)

Author: Niraj & Senior Analytics Mentor
Purpose: Generates a reproducible raw shipment dataset simulating realistic
         operational delays across US trade routes.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Configuration & Reproducibility
# ---------------------------------------------------------
SEED = 42
N_RECORDS = 5000
rng = np.random.default_rng(SEED)

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "shipments_raw.csv"


def generate_logistics_data(n: int = N_RECORDS) -> pd.DataFrame:
    """Generates synthetic US logistics shipment records based on predefined

    operational simulation rules.
    """
    # ---------------------------------------------------------
    # 2. Sequential Identifiers
    # ---------------------------------------------------------
    order_ids = [f"ORD-{i:05d}" for i in range(1, n + 1)]

    # ---------------------------------------------------------
    # 3. Order Dates (Q1 2024 with explicit Day-of-Week Weights)
    # ---------------------------------------------------------
    # Generate all days between 2024-01-01 and 2024-03-31 (91 calendar days)
    calendar_dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="D")

    # Day-of-week weights: Mon=0, Tue=1, ..., Sun=6
    # Monday (22%) and Tuesday (20%) have higher e-commerce order volumes
    day_weights = {
        0: 0.22,  # Monday
        1: 0.20,  # Tuesday
        2: 0.16,  # Wednesday
        3: 0.15,  # Thursday
        4: 0.13,  # Friday
        5: 0.07,  # Saturday
        6: 0.07,  # Sunday
    }

    date_probs = np.array([day_weights[d.weekday()] for d in calendar_dates])
    date_probs = date_probs / date_probs.sum()  # Normalize to sum exactly to 1.0

    selected_date_indices = rng.choice(len(calendar_dates), size=n, p=date_probs)
    order_dates = calendar_dates[selected_date_indices]
    # Sort order dates chronologically so the log reads realistically
    order_dates = pd.DatetimeIndex(np.sort(order_dates))

    # ---------------------------------------------------------
    # 4. Warehouses & Destination Markets (US-based)
    # ---------------------------------------------------------
    warehouses = ["WH-East", "WH-West", "WH-North", "WH-South"]
    wh_probs = [0.35, 0.25, 0.20, 0.20]
    origin_warehouses = rng.choice(warehouses, size=n, p=wh_probs)

    destinations = ["New York", "Los Angeles", "Chicago", "Dallas", "Atlanta"]
    dest_probs = [0.30, 0.25, 0.18, 0.15, 0.12]
    destination_cities = rng.choice(destinations, size=n, p=dest_probs)

    # ---------------------------------------------------------
    # 5. Route Distance (KM) Lookup Table + Random Deviation
    # ---------------------------------------------------------
    # Approximate highway distances between regional hub locations
    distance_map = {
        ("WH-East", "New York"): 150,
        ("WH-East", "Los Angeles"): 4000,
        ("WH-East", "Chicago"): 1300,
        ("WH-East", "Dallas"): 2400,
        ("WH-East", "Atlanta"): 1200,
        ("WH-West", "New York"): 3950,
        ("WH-West", "Los Angeles"): 120,
        ("WH-West", "Chicago"): 2800,
        ("WH-West", "Dallas"): 2250,
        ("WH-West", "Atlanta"): 3450,
        ("WH-North", "New York"): 1300,
        ("WH-North", "Los Angeles"): 2800,
        ("WH-North", "Chicago"): 90,
        ("WH-North", "Dallas"): 1500,
        ("WH-North", "Atlanta"): 1150,
        ("WH-South", "New York"): 2400,
        ("WH-South", "Los Angeles"): 2250,
        ("WH-South", "Chicago"): 1500,
        ("WH-South", "Dallas"): 80,
        ("WH-South", "Atlanta"): 1250,
    }

    base_distances = np.array(
        [
            distance_map[(orig, dest)]
            for orig, dest in zip(origin_warehouses, destination_cities)
        ],
        dtype=float,
    )
    # Add small route variance (+/- 25 km)
    distance_km = np.round(
        base_distances + rng.uniform(-25.0, 25.0, size=n), 1
    )
    distance_km = np.maximum(distance_km, 50.0)  # Floor at 50 km minimum

    # ---------------------------------------------------------
    # 6. Package Weight (Right-Skewed Distribution)
    # ---------------------------------------------------------
    # Most packages are light consumer parcels; heavy packages (>25 kg) are infrequent
    raw_weights = rng.exponential(scale=6.0, size=n) + 0.5
    package_weights = np.round(np.clip(raw_weights, 0.5, 45.0), 2)

    # ---------------------------------------------------------
    # 7. Carrier Selection
    # ---------------------------------------------------------
    carriers_list = ["SwiftCargo", "FastTrack", "ExpressAir"]
    carrier_probs = [0.45, 0.35, 0.20]
    carriers = rng.choice(carriers_list, size=n, p=carrier_probs)

    # ---------------------------------------------------------
    # 8. Weather & Traffic Conditions
    # ---------------------------------------------------------
    weathers_list = ["Clear", "Rain", "Fog", "Storm"]
    weather_probs = [0.70, 0.20, 0.07, 0.03]
    weathers = rng.choice(weathers_list, size=n, p=weather_probs)

    traffics_list = ["Low", "Medium", "High"]
    traffic_probs = [0.30, 0.50, 0.20]
    traffics = rng.choice(traffics_list, size=n, p=traffic_probs)

    # ---------------------------------------------------------
    # 9. Operational Rules: Warehouse Processing Days (Delay Rule 1)
    # ---------------------------------------------------------
    # Baseline: 80% take 1 day, 20% take 2 days
    proc_days = rng.choice([1, 2], size=n, p=[0.80, 0.20])

    # Bottleneck A: WH-North has a 50% chance of adding +1 processing day
    wh_north_penalty = (origin_warehouses == "WH-North") & (rng.random(size=n) < 0.50)
    proc_days += wh_north_penalty.astype(int)

    # Bottleneck B: Heavy packages (>25 kg) require manual team handling (40% chance of +1 day)
    heavy_pkg_penalty = (package_weights > 25.0) & (rng.random(size=n) < 0.40)
    proc_days += heavy_pkg_penalty.astype(int)

    # Ship Date = Order Date + Processing Days
    ship_dates = order_dates + pd.to_timedelta(proc_days, unit="D")

    # ---------------------------------------------------------
    # 10. Operational Rules: Promised Delivery Date (Customer SLA)
    # ---------------------------------------------------------
    # Standard customer SLA window based on distance
    sla_days = np.where(
        distance_km < 600,
        3,
        np.where(distance_km <= 1800, 5, 7),
    )

    # ExpressAir promises 2 days faster (floored at minimum 2 days)
    express_mask = carriers == "ExpressAir"
    sla_days[express_mask] = np.maximum(2, sla_days[express_mask] - 2)

    promised_dates = order_dates + pd.to_timedelta(sla_days, unit="D")

    # ---------------------------------------------------------
    # 11. Operational Rules: Actual Transit Days (Delay Rules 2-5)
    # ---------------------------------------------------------
    # Base transit time roughly 650 km per driving day
    base_transit = np.maximum(1, np.round(distance_km / 650.0).astype(int))

    # Delay Rule 2 (Carrier):
    # ExpressAir transit is 1 day faster on long routes
    carrier_transit_mod = np.zeros(n, dtype=int)
    carrier_transit_mod[express_mask & (base_transit > 1)] -= 1

    # SwiftCargo (budget) has a 25% chance of +1 day mechanical/transit lag
    swift_lag = (carriers == "SwiftCargo") & (rng.random(size=n) < 0.25)
    carrier_transit_mod += swift_lag.astype(int)

    # Delay Rule 3 (Weather):
    # Storms add +2 days; Rain/Fog have 20% chance of adding +1 day
    weather_penalty = np.zeros(n, dtype=int)
    weather_penalty[weathers == "Storm"] += 2
    mild_weather_mask = np.isin(weathers, ["Rain", "Fog"]) & (rng.random(size=n) < 0.20)
    weather_penalty[mild_weather_mask] += 1

    # Delay Rule 4 (Traffic):
    # High traffic has 35% chance of adding +1 day
    traffic_penalty = ((traffics == "High") & (rng.random(size=n) < 0.35)).astype(int)

    # Delay Rule 5 (Heavy Parcel Transit Friction):
    # Parcels > 25 kg have 20% chance of taking +1 extra day in transit
    weight_transit_penalty = ((package_weights > 25.0) & (rng.random(size=n) < 0.20)).astype(int)

    total_transit_days = np.maximum(
        1,
        base_transit
        + carrier_transit_mod
        + weather_penalty
        + traffic_penalty
        + weight_transit_penalty,
    )

    actual_dates = ship_dates + pd.to_timedelta(total_transit_days, unit="D")

    # ---------------------------------------------------------
    # 12. Derived Business Metrics
    # ---------------------------------------------------------
    delay_days = (actual_dates - promised_dates).days.to_numpy()
    delayed = (delay_days > 0).astype(int)

    # ---------------------------------------------------------
    # 13. Assemble DataFrame
    # ---------------------------------------------------------
    df = pd.DataFrame(
        {
            "Order_ID": order_ids,
            "Order_Date": order_dates.strftime("%Y-%m-%d"),
            "Origin_Warehouse": origin_warehouses,
            "Destination_City": destination_cities,
            "Distance_KM": distance_km,
            "Package_Weight_KG": package_weights,
            "Carrier": carriers,
            "Weather_Condition": weathers,
            "Traffic_Level": traffics,
            "Warehouse_Processing_Days": proc_days,
            "Ship_Date": ship_dates.strftime("%Y-%m-%d"),
            "Promised_Delivery_Date": promised_dates.strftime("%Y-%m-%d"),
            "Transit_Days": total_transit_days,
            "Actual_Delivery_Date": actual_dates.strftime("%Y-%m-%d"),
            "Delay_Days": delay_days,
            "Delayed": delayed,
        }
    )

    return df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_logistics_data(N_RECORDS)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset successfully generated and saved to: {OUTPUT_FILE}")
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
