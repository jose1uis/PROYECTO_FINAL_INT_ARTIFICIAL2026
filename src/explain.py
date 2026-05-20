from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def get_transformed_data(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    transformed = pipeline.named_steps["preprocessor"].transform(x)
    return np.asarray(transformed)


def explain_student(
    pipeline: Pipeline,
    x_background: pd.DataFrame,
    student_row: pd.DataFrame,
    feature_names: list[str],
    max_features: int = 8,
) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    background_transformed = get_transformed_data(pipeline, x_background)
    row_transformed = get_transformed_data(pipeline, student_row)

    explainer = shap.TreeExplainer(model, background_transformed)
    shap_values = explainer.shap_values(row_transformed)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]
        if values.ndim == 2:
            values = values[:, 1]

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "value": row_transformed[0],
            "impact": values,
            "abs_impact": np.abs(values),
        }
    )
    return explanation.sort_values("abs_impact", ascending=False).head(max_features)


def summarize_student(row: pd.Series, top_factors: pd.DataFrame) -> dict:
    return {
        "student_profile": row.to_dict(),
        "top_factors": top_factors[["feature", "value", "impact"]].to_dict(orient="records"),
    }

