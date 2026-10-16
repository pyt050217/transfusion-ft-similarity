import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Config
# ============================================================

SEED = 42
RARE_MIN_COUNT = 20

DATA_PATH = "/root/transfusion/data.csv"
OUT_DIR = Path("/root/transfusion/processed_ft")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

random.seed(SEED)
np.random.seed(SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Column names
# ============================================================

REG_ID = "\u767b\u8bb0\u53f7"
VISIT_ID = "\u5c31\u8bca\u53f7"

SEX = "\u6027\u522b"
DEPT = "\u5165\u9662\u79d1\u5ba4"
SURGERY_TYPE = "\u624b\u672f\u7c7b\u578b"
ASA = "\u9ebb\u9189\u5206\u7ea7"
DIAG = "\u672f\u524d\u4e3b\u8bca\u65ad\u7f16\u7801"

AGE = "\u5e74\u9f84"
HEIGHT = "\u8eab\u9ad8"
WEIGHT = "\u4f53\u91cd"

PRE = "\u4e34\u5e8a\u672f\u524d_"
INTRA = "\u4e34\u5e8a\u672f\u4e2d_"

RBC = "\u7ea2\u7ec6\u80de"

TARGET = INTRA + RBC

CATEGORICAL_COLS = [
    SEX,
    DEPT,
    SURGERY_TYPE,
    ASA,
    DIAG,
]

CONTINUOUS_COLS = [
    AGE,
    HEIGHT,
    WEIGHT,
    PRE + "HGB",
    PRE + "HCT",
    PRE + "PLT",
    PRE + "WBC",
    PRE + "APTT",
    PRE + "PT",
    PRE + "FIB",
    PRE + "CREA",
    PRE + "ALB",
    PRE + "eGFR",
]

USECOLS = (
    [REG_ID, VISIT_ID]
    + CATEGORICAL_COLS
    + CONTINUOUS_COLS
    + [TARGET]
)


# ============================================================
# First-pass hard validity ranges
# Only obvious impossible / highly implausible values are removed.
# Pediatric values are intentionally retained.
# ============================================================

HARD_RANGES = {
    AGE: (0, 120),
    HEIGHT: (20, 250),
    WEIGHT: (1, 400),
    PRE + "HGB": (20, 250),
    PRE + "HCT": (0.05, 0.70),
    PRE + "PLT": (1, 2000),
    PRE + "WBC": (0.1, 100),
    PRE + "APTT": (5, 300),
    PRE + "PT": (5, 100),
    PRE + "FIB": (0.1, 15),
    PRE + "CREA": (5, 3000),
    PRE + "ALB": (5, 70),
    PRE + "eGFR": (1, 700),
}


# ============================================================
# Helpers
# ============================================================

def clean_string_series(s):
    s = s.astype("string")
    s = s.str.strip()

    bad = s.isna() | (s == "") | (s.str.lower() == "nan")
    s = s.mask(bad)

    return s


def clean_categories(df):
    for col in CATEGORICAL_COLS:
        df[col] = clean_string_series(df[col])

    # Surgery type:
    # "elective," -> "elective"
    df[SURGERY_TYPE] = (
        df[SURGERY_TYPE]
        .str.rstrip(",")
        .str.strip()
    )

    # One extremely rare invalid value was observed in audit.
    valid_surgery = {
        "\u62e9\u671f\u624b\u672f",
        "\u65e5\u95f4\u624b\u672f",
        "\u6025\u8bca\u624b\u672f",
    }

    df[SURGERY_TYPE] = df[SURGERY_TYPE].where(
        df[SURGERY_TYPE].isin(valid_surgery)
    )

    # Normalize ASA:
    # 2, / 2.0 / "2" -> "2"
    asa_numeric = pd.to_numeric(
        df[ASA].str.rstrip(","),
        errors="coerce"
    )

    asa_numeric = asa_numeric.where(
        asa_numeric.between(1, 6)
    )

    df[ASA] = asa_numeric.map(
        lambda x: str(int(x)) if pd.notna(x) else pd.NA
    )

    return df


def clean_numeric(df):
    report = {}

    for col in CONTINUOUS_COLS:
        x = pd.to_numeric(df[col], errors="coerce")

        before_valid = int(x.notna().sum())

        if col in HARD_RANGES:
            lo, hi = HARD_RANGES[col]
            invalid = x.notna() & ((x < lo) | (x > hi))
            removed = int(invalid.sum())
            x = x.mask(invalid)
        else:
            removed = 0

        df[col] = x

        report[col] = {
            "valid_before": before_valid,
            "removed_by_range": removed,
            "missing_after": int(x.isna().sum()),
        }

    return df, report


def make_patient_split(df):
    # One row per patient for splitting.
    # If any surgery for that patient had intraoperative RBC > 0,
    # patient-level label is positive.
    patient_df = (
        df.groupby(REG_ID, as_index=False)["target_binary"]
        .max()
    )

    train_patients, temp_patients = train_test_split(
        patient_df,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=patient_df["target_binary"],
    )

    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=relative_test_ratio,
        random_state=SEED,
        stratify=temp_patients["target_binary"],
    )

    train_ids = set(train_patients[REG_ID])
    val_ids = set(val_patients[REG_ID])
    test_ids = set(test_patients[REG_ID])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    train_df = df[df[REG_ID].isin(train_ids)].copy()
    val_df = df[df[REG_ID].isin(val_ids)].copy()
    test_df = df[df[REG_ID].isin(test_ids)].copy()

    return train_df, val_df, test_df


def build_category_metadata(train_df):
    metadata = {}

    for col in CATEGORICAL_COLS:
        s = train_df[col]

        counts = s.value_counts(dropna=True)

        frequent = sorted(
            counts[counts >= RARE_MIN_COUNT]
            .index
            .astype(str)
            .tolist()
        )

        # Reserved:
        # 0 = UNKNOWN
        # 1 = MISSING
        # 2 = RARE
        vocab = {
            "__UNKNOWN__": 0,
            "__MISSING__": 1,
            "__RARE__": 2,
        }

        for value in frequent:
            vocab[value] = len(vocab)

        metadata[col] = {
            "vocab": vocab,
            "rare_min_count": RARE_MIN_COUNT,
        }

    return metadata


def encode_category(series, metadata):
    vocab = metadata["vocab"]

    def encode_one(value):
        if pd.isna(value):
            return vocab["__MISSING__"]

        value = str(value)

        if value in vocab:
            return vocab[value]

        return vocab["__RARE__"]

    return series.map(encode_one).astype(np.int64)


def build_numeric_metadata(train_df):
    metadata = {}

    for col in CONTINUOUS_COLS:
        x = pd.to_numeric(train_df[col], errors="coerce")

        median = float(x.median())

        if not np.isfinite(median):
            median = 0.0

        filled = x.fillna(median)

        mean = float(filled.mean())
        std = float(filled.std())

        if (not np.isfinite(std)) or std < 1e-8:
            std = 1.0

        metadata[col] = {
            "median": median,
            "mean": mean,
            "std": std,
        }

    return metadata


def transform_split(
    frame,
    category_meta,
    numeric_meta,
):
    # Categorical
    x_cat = np.stack(
        [
            encode_category(
                frame[col],
                category_meta[col]
            ).to_numpy()
            for col in CATEGORICAL_COLS
        ],
        axis=1,
    ).astype(np.int64)

    # Continuous + missing indicators
    numerical_parts = []

    for col in CONTINUOUS_COLS:
        x = pd.to_numeric(
            frame[col],
            errors="coerce"
        )

        missing = x.isna().astype(np.float32).to_numpy()

        median = numeric_meta[col]["median"]
        mean = numeric_meta[col]["mean"]
        std = numeric_meta[col]["std"]

        x = x.fillna(median)
        x = ((x - mean) / std).astype(np.float32).to_numpy()

        numerical_parts.append(x)
        numerical_parts.append(missing)

    x_cont = np.stack(
        numerical_parts,
        axis=1,
    ).astype(np.float32)

    y_binary = frame["target_binary"].to_numpy(
        dtype=np.int64
    )

    y_amount = frame["target_amount"].to_numpy(
        dtype=np.float32
    )

    return x_cat, x_cont, y_binary, y_amount


def save_split(
    name,
    frame,
    x_cat,
    x_cont,
    y_binary,
    y_amount,
):
    np.savez_compressed(
        OUT_DIR / f"{name}.npz",
        x_categ=x_cat,
        x_cont=x_cont,
        y_binary=y_binary,
        y_amount=y_amount,
    )

    # Keep row-level identity separately for future retrieval.
    index_df = frame[
        [REG_ID, VISIT_ID]
    ].copy()

    index_df["target_binary"] = y_binary
    index_df["target_amount"] = y_amount

    index_df.to_csv(
        OUT_DIR / f"{name}_index.csv",
        index=False,
        encoding="utf-8-sig",
    )


def print_split_stats(name, frame):
    print(f"\n[{name}]")
    print("Rows:", len(frame))
    print(
        "Patients:",
        frame[REG_ID].nunique()
    )
    print(
        "Positive rows:",
        int(frame["target_binary"].sum())
    )
    print(
        "Positive row rate:",
        f'{frame["target_binary"].mean() * 100:.4f}%'
    )


# ============================================================
# Load
# ============================================================

print("Loading data...")

df = pd.read_csv(
    DATA_PATH,
    encoding="gb18030",
    usecols=USECOLS,
    low_memory=False,
)

print("Raw rows:", len(df))
print("Raw patients:", df[REG_ID].nunique())


# ============================================================
# Target
# ============================================================

target = pd.to_numeric(
    df[TARGET],
    errors="coerce"
).fillna(0)

# Negative blood amounts should never occur.
target = target.clip(lower=0)

df["target_amount"] = target.astype(np.float32)
df["target_binary"] = (target > 0).astype(np.int64)


# ============================================================
# Clean
# ============================================================

print("\nCleaning categorical variables...")
df = clean_categories(df)

print("Cleaning numeric variables...")
df, numeric_clean_report = clean_numeric(df)


# ============================================================
# Patient-level split
# ============================================================

print("\nCreating patient-level split...")

train_df, val_df, test_df = make_patient_split(df)

print_split_stats("TRAIN", train_df)
print_split_stats("VAL", val_df)
print_split_stats("TEST", test_df)


# ============================================================
# Fit preprocessing on TRAIN ONLY
# ============================================================

print("\nFitting categorical metadata from TRAIN only...")
category_meta = build_category_metadata(train_df)

print("Fitting numeric metadata from TRAIN only...")
numeric_meta = build_numeric_metadata(train_df)


# ============================================================
# Transform
# ============================================================

print("\nTransforming TRAIN...")
train_arrays = transform_split(
    train_df,
    category_meta,
    numeric_meta
)

print("Transforming VAL...")
val_arrays = transform_split(
    val_df,
    category_meta,
    numeric_meta
)

print("Transforming TEST...")
test_arrays = transform_split(
    test_df,
    category_meta,
    numeric_meta
)


# ============================================================
# Save arrays
# ============================================================

print("\nSaving processed datasets...")

save_split(
    "train",
    train_df,
    *train_arrays
)

save_split(
    "val",
    val_df,
    *val_arrays
)

save_split(
    "test",
    test_df,
    *test_arrays
)


# ============================================================
# Metadata
# ============================================================

category_sizes = [
    len(category_meta[col]["vocab"])
    for col in CATEGORICAL_COLS
]

metadata = {
    "seed": SEED,
    "rare_min_count": RARE_MIN_COUNT,
    "categorical_cols": CATEGORICAL_COLS,
    "continuous_cols": CONTINUOUS_COLS,
    "continuous_feature_layout": [
        item
        for col in CONTINUOUS_COLS
        for item in [
            col,
            col + "__missing",
        ]
    ],
    "category_sizes": category_sizes,
    "category_metadata": category_meta,
    "numeric_metadata": numeric_meta,
    "numeric_clean_report": numeric_clean_report,
    "target_col": TARGET,
    "target_definition": "intraoperative RBC > 0",
    "split_method": "patient-level stratified split by registration ID",
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
}

with open(
    OUT_DIR / "preprocessing_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Final checks
# ============================================================

train_patient_ids = set(train_df[REG_ID])
val_patient_ids = set(val_df[REG_ID])
test_patient_ids = set(test_df[REG_ID])

assert train_patient_ids.isdisjoint(val_patient_ids)
assert train_patient_ids.isdisjoint(test_patient_ids)
assert val_patient_ids.isdisjoint(test_patient_ids)

print("\nCategory sizes:")
for col, size in zip(
    CATEGORICAL_COLS,
    category_sizes
):
    print(f"{col}: {size}")

print("\nArray shapes:")

for name, arrays in [
    ("train", train_arrays),
    ("val", val_arrays),
    ("test", test_arrays),
]:
    x_cat, x_cont, y_binary, y_amount = arrays

    print(
        name,
        "x_categ =", x_cat.shape,
        "x_cont =", x_cont.shape,
        "y =", y_binary.shape,
    )

print("\nOutput directory:")
print(OUT_DIR)

print("\nPreprocessing finished successfully.")
