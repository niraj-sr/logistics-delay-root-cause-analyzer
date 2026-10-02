"""
Logistics Delay Root-Cause Analyzer
Data Cleaning & Validation Pipeline

Author: Niraj & Senior Analytics Mentor
Purpose: Loads raw logistics shipment data, enforces strict typing,
         executes defensive data validation checks, and saves a verified
         clean dataset for exploratory analysis.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# 1. File Paths Configuration
# ---------------------------------------------------------
RAW_DATA_PATH = Path("data/raw/shipments_raw.csv")
PROCESSED_DIR = Path("data/processed")
CLEANED_DATA_PATH = PROCESSED_DIR / "shipments_cleaned.csv"

# ---------------------------------------------------------
# 2. Approved Categorical Domains (Reference Schema)
# ---------------------------------------------------------
VALID_WAREHOUSES = {"WH-East", "WH-West", "WH-North", "WH-South"}
VALID_CITIES = {"New York", "Los Angeles", "Chicago", "Dallas", "Atlanta"}
VALID_CARRIERS = {"SwiftCargo", "FastTrack", "ExpressAir"}
VALID_WEATHERS = {"Clear", "Rain", "Fog", "Storm"}
VALID_TRAFFICS = {"Low", "Medium", "High"}

DATE_COLUMNS = [
    "Order_Date",
    "Ship_Date",
    "Promised_Delivery_Date",
    "Actual_Delivery_Date",
]


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Loads raw shipment data and parses date columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Convert date columns to native pandas datetime64
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col])

    return df


def validate_shipment_data(df: pd.DataFrame) -> dict:
    """Executes defensive validation checks across data integrity dimensions.

    Raises ValueError if any rule is violated.
    """
    errors = []

    # 1. Missing Values Check
    total_nulls = int(df.isnull().sum().sum())
    if total_nulls > 0:
        null_breakdown = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
        errors.append(f"Missing values detected: {null_breakdown}")

    # 2. Categorical Integrity Checks
    invalid_wh = set(df["Origin_Warehouse"].unique()) - VALID_WAREHOUSES
    invalid_city = set(df["Destination_City"].unique()) - VALID_CITIES
    invalid_carrier = set(df["Carrier"].unique()) - VALID_CARRIERS
    invalid_weather = set(df["Weather_Condition"].unique()) - VALID_WEATHERS
    invalid_traffic = set(df["Traffic_Level"].unique()) - VALID_TRAFFICS

    cat_violations = len(invalid_wh) + len(invalid_city) + len(invalid_carrier) + len(invalid_weather) + len(invalid_traffic)

    if cat_violations > 0:
        errors.append(
            f"Invalid categories found: Warehouses={invalid_wh}, Cities={invalid_city}, "
            f"Carriers={invalid_carrier}, Weather={invalid_weather}, Traffic={invalid_traffic}"
        )

    # 3. Numerical Boundary Checks
    non_positive_distance = int((df["Distance_KM"] <= 0).sum())
    non_positive_weight = int((df["Package_Weight_KG"] <= 0).sum())
    num_violations = non_positive_distance + non_positive_weight

    if num_violations > 0:
        errors.append(
            f"Numerical boundary violations: Non-positive distance={non_positive_distance}, "
            f"Non-positive weight={non_positive_weight}"
        )

    # 4. Chronological Sequence Checks
    ship_before_order = int((df["Ship_Date"] < df["Order_Date"]).sum())
    actual_before_ship = int((df["Actual_Delivery_Date"] < df["Ship_Date"]).sum())
    date_violations = ship_before_order + actual_before_ship

    if date_violations > 0:
        errors.append(
            f"Chronological violations: Ship < Order={ship_before_order}, "
            f"Actual Delivery < Ship={actual_before_ship}"
        )

    # If any integrity check failed, halt execution
    if errors:
        error_message = "\n".join([f"- {err}" for err in errors])
        raise ValueError(f"Data Validation Failed:\n{error_message}")

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "total_nulls": total_nulls,
        "date_violations": date_violations,
        "categorical_violations": cat_violations,
        "numerical_violations": num_violations,
    }


def save_cleaned_data(df: pd.DataFrame, output_path: Path) -> None:
    """Exports validated clean dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_pipeline() -> None:
    """Main execution flow for data cleaning and validation."""
    print("=" * 60)
    print("LOGISTICS DATA CLEANING & VALIDATION PIPELINE")
    print("=" * 60)
    print(f"Reading raw data from: {RAW_DATA_PATH} ...")

    df = load_raw_data(RAW_DATA_PATH)

    print("Executing defensive validation checks ...")
    summary = validate_shipment_data(df)

    print("All validation checks PASSED (0 violations detected).")
    print(f"Saving cleaned dataset to: {CLEANED_DATA_PATH} ...")
    save_cleaned_data(df, CLEANED_DATA_PATH)

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Rows Processed:       {summary['rows']}")
    print(f"Total Columns:              {summary['columns']}")
    print(f"Total Null / Missing:       {summary['total_nulls']}")
    print(f"Chronological Violations:   {summary['date_violations']}")
    print(f"Categorical Violations:     {summary['categorical_violations']}")
    print(f"Numerical Violations:       {summary['numerical_violations']}")
    print(f"Cleaned Output File:        {CLEANED_DATA_PATH.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
