# Asistente IA para alerta temprana de desercion estudiantil

Proyecto final de Inteligencia Artificial: predice riesgo de desercion universitaria con Machine Learning, explica los factores principales con SHAP y genera recomendaciones en lenguaje natural usando la API de OpenAI.

## Componentes

- EDA y entrenamiento de un modelo de clasificacion.
- Pipeline reproducible con `train/test split`.
- Metricas: precision, recall, F1, ROC-AUC y matriz de confusion.
- Explicabilidad con SHAP para cada estudiante.
- Interfaz en Streamlit.
- Recomendaciones automaticas con OpenAI.

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configurar OpenAI

Crea un archivo `.env` en la raiz del proyecto:

```bash
OPENAI_API_KEY=tu_api_key
OPENAI_MODEL=gpt-5
```

Si no configuras la API key, la app igual funciona, pero las recomendaciones se generan con una plantilla local.

## Ejecutar la app

```bash
streamlit run app.py
```

## Dataset recomendado

Puedes usar el dataset de Kaggle/UCI sobre prediccion de abandono estudiantil:

- Higher Education Predictors of Student Retention
- Student Dropout Prediction

La app tambien incluye un dataset demo sintetico para que puedas probar todo de inmediato.

## Estructura

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
├── scripts/
│   └── make_demo_data.py
└── src/
    ├── advisor.py
    ├── data_utils.py
    ├── explain.py
    └── modeling.py
```

## Como usarlo en la entrega

1. Corre la app con el dataset demo.
2. Descarga el dataset real y cargalo desde la barra lateral.
3. Entrena el modelo.
4. Muestra las metricas y la explicacion de un estudiante.
5. Genera una recomendacion con OpenAI.
6. Graba un video demo de maximo 3 minutos.

