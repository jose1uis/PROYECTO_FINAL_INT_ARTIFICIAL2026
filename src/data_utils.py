from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    target_column: str
    positive_label: str | int | float


def make_demo_dataset(n_rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic student dropout dataset for demos."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(17, 45, n_rows)
    admission_grade = rng.normal(70, 12, n_rows).clip(0, 100)
    first_semester_grade = rng.normal(3.3, 0.7, n_rows).clip(0, 5)
    approved_credits = rng.integers(0, 22, n_rows)
    failed_subjects = rng.poisson(1.2, n_rows).clip(0, 8)
    tuition_up_to_date = rng.choice(["yes", "no"], n_rows, p=[0.82, 0.18])
    scholarship = rng.choice(["yes", "no"], n_rows, p=[0.28, 0.72])
    attendance_rate = rng.normal(0.82, 0.14, n_rows).clip(0, 1)
    commute_time = rng.choice(["short", "medium", "long"], n_rows, p=[0.46, 0.36, 0.18])
    program = rng.choice(
        ["Engineering", "Business", "Social Sciences", "Health", "Design"],
        n_rows,
        p=[0.32, 0.24, 0.18, 0.16, 0.10],
    )

    risk_score = (
        -2.2
        + 0.08 * failed_subjects
        - 0.70 * first_semester_grade
        - 0.035 * approved_credits
        - 1.15 * attendance_rate
        + 0.42 * (tuition_up_to_date == "no")
        - 0.28 * (scholarship == "yes")
        + 0.34 * (commute_time == "long")
        + 0.018 * (age - 23)
        - 0.006 * admission_grade
    )
    probability = 1 / (1 + np.exp(-risk_score))
    dropout = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "age": age,
            "admission_grade": admission_grade.round(2),
            "first_semester_grade": first_semester_grade.round(2),
            "approved_credits": approved_credits,
            "failed_subjects": failed_subjects,
            "tuition_up_to_date": tuition_up_to_date,
            "scholarship": scholarship,
            "attendance_rate": attendance_rate.round(2),
            "commute_time": commute_time,
            "program": program,
            "dropout": np.where(dropout == 1, "Dropout", "Not dropout"),
        }
    )


def infer_target_column(columns: Iterable[str]) -> str | None:
    candidates = ["dropout", "target", "status", "outcome", "desercion", "abandono"]
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def prepare_target(series: pd.Series, positive_label: str | int | float) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (series == positive_label).astype(int)

    positive = str(positive_label).strip().lower()
    return series.astype(str).str.strip().str.lower().eq(positive).astype(int)


def split_features_target(df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.Series]:
    clean_df = df.dropna(axis=0, how="all").copy()
    y = prepare_target(clean_df[config.target_column], config.positive_label)
    x = clean_df.drop(columns=[config.target_column])
    return x, y

