from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def build_local_recommendation(probability: float, top_factors: list[dict]) -> str:
    risk = "alto" if probability >= 0.65 else "medio" if probability >= 0.35 else "bajo"
    factor_names = ", ".join(str(item["feature"]) for item in top_factors[:4])
    return (
        f"Riesgo estimado: {risk} ({probability:.1%}). "
        f"Los factores mas importantes detectados fueron: {factor_names}. "
        "Se recomienda revisar acompanamiento academico, apoyo financiero, seguimiento de asistencia "
        "y una entrevista breve para validar causas personales o institucionales."
    )


def generate_recommendation(probability: float, student_summary: dict) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return build_local_recommendation(probability, student_summary["top_factors"])

    model = os.getenv("OPENAI_MODEL", "gpt-5")
    client = OpenAI(api_key=api_key)

    prompt = f"""
Eres un analista academico de una universidad. Debes explicar el riesgo de desercion de un estudiante
de forma clara, responsable y accionable. No inventes datos. No menciones SHAP si no es necesario.

Probabilidad estimada de desercion: {probability:.3f}

Perfil y factores principales:
{student_summary}

Entrega:
1. Nivel de riesgo en una frase.
2. Explicacion breve de los factores principales.
3. Tres recomendaciones concretas para intervencion academica o bienestar.
4. Una nota de cautela indicando que el modelo apoya decisiones, pero no reemplaza evaluacion humana.
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text

