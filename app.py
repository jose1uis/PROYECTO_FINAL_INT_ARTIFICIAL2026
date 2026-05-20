from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.advisor import generate_recommendation
from src.data_utils import DatasetConfig, infer_target_column, make_demo_dataset, split_features_target
from src.explain import explain_student, summarize_student
from src.modeling import train_model


st.set_page_config(
    page_title="Alerta temprana de desercion",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    return make_demo_dataset()


@st.cache_data
def load_uploaded_csv(file) -> pd.DataFrame:
    return pd.read_csv(file, sep=None, engine="python")


@st.cache_resource
def cached_training(df: pd.DataFrame, target_column: str, positive_label: str):
    x, y = split_features_target(df, DatasetConfig(target_column, positive_label))
    return train_model(x, y)


def metric_card(label: str, value: float) -> None:
    st.metric(label, f"{value:.3f}")


st.title("Asistente IA para alerta temprana de desercion estudiantil")
st.caption("Machine Learning + SHAP + recomendaciones generadas con OpenAI")

with st.sidebar:
    st.header("Datos")
    uploaded = st.file_uploader("Carga un CSV", type=["csv"])
    if uploaded:
        df = load_uploaded_csv(uploaded)
    else:
        df = load_demo_data()
        st.info("Usando dataset demo sintetico.")

    inferred_target = infer_target_column(df.columns)
    target_index = list(df.columns).index(inferred_target) if inferred_target in df.columns else len(df.columns) - 1
    target_column = st.selectbox("Variable objetivo", df.columns, index=target_index)

    labels = sorted(df[target_column].dropna().unique().tolist(), key=lambda item: str(item))
    default_positive_index = 0
    for idx, label in enumerate(labels):
        label_text = str(label).lower()
        if "dropout" in label_text or "desert" in label_text or "aband" in label_text:
            default_positive_index = idx
            break
    positive_label = st.selectbox("Etiqueta de riesgo/desercion", labels, index=default_positive_index)

    st.divider()
    st.write("Filas:", len(df))
    st.write("Columnas:", len(df.columns))

if df.empty:
    st.error("El dataset esta vacio.")
    st.stop()

tab_data, tab_model, tab_student = st.tabs(["Datos", "Modelo", "Estudiante"])

with tab_data:
    st.subheader("Vista previa")
    st.dataframe(df.head(30), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        target_counts = df[target_column].astype(str).value_counts().reset_index()
        target_counts.columns = [target_column, "count"]
        fig = px.bar(target_counts, x=target_column, y="count", title="Distribucion de la variable objetivo")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        if numeric_columns:
            selected_numeric = st.selectbox("Variable numerica para explorar", numeric_columns)
            fig = px.histogram(df, x=selected_numeric, color=df[target_column].astype(str), marginal="box")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay columnas numericas para graficar.")

with tab_model:
    st.subheader("Entrenamiento y evaluacion")
    result = cached_training(df, target_column, positive_label)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Accuracy", result.metrics["accuracy"])
    with col2:
        metric_card("Precision", result.metrics["precision"])
    with col3:
        metric_card("Recall", result.metrics["recall"])
    with col4:
        metric_card("F1", result.metrics["f1"])
    with col5:
        metric_card("ROC-AUC", result.metrics["roc_auc"])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Matriz de confusion")
        confusion_df = pd.DataFrame(
            result.confusion,
            index=["Real: no riesgo", "Real: riesgo"],
            columns=["Pred: no riesgo", "Pred: riesgo"],
        )
        st.dataframe(confusion_df, use_container_width=True)

    with col2:
        importance = pd.DataFrame(
            {
                "feature": result.feature_names,
                "importance": result.pipeline.named_steps["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False).head(15)
        fig = px.bar(importance, x="importance", y="feature", orientation="h", title="Importancia global")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

with tab_student:
    st.subheader("Explicacion individual y recomendacion")
    result = cached_training(df, target_column, positive_label)

    selected_index = st.slider("Selecciona un estudiante del conjunto de prueba", 0, len(result.x_test) - 1, 0)
    student_row = result.x_test.iloc[[selected_index]]
    probability = result.pipeline.predict_proba(student_row)[0, 1]
    predicted_class = int(probability >= 0.5)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Probabilidad de desercion", f"{probability:.1%}")
        st.write("Clasificacion:", "Riesgo" if predicted_class == 1 else "No riesgo")
        st.dataframe(student_row.T.rename(columns={student_row.index[0]: "valor"}), use_container_width=True)

    with col2:
        background = result.x_train.sample(min(150, len(result.x_train)), random_state=42)
        top_factors = explain_student(
            result.pipeline,
            background,
            student_row,
            result.feature_names,
        )
        fig = px.bar(
            top_factors.sort_values("impact"),
            x="impact",
            y="feature",
            orientation="h",
            title="Factores que mas influyen en la prediccion",
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Generar recomendacion con OpenAI", type="primary"):
        summary = summarize_student(student_row.iloc[0], top_factors)
        with st.spinner("Generando recomendacion..."):
            recommendation = generate_recommendation(probability, summary)
        st.markdown("### Recomendacion")
        st.write(recommendation)
