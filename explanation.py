from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter()

class ExplanationRequest(BaseModel):
    temperature: float = Field(..., ge=-50, le=100, description="Temperatura în grade Celsius")
    current: float = Field(..., ge=0, description="Curentul în amperi")
    battery_soc: float = Field(..., ge=0, le=100, description="Nivelul de încărcare al bateriei în %")
    deviation: Optional[float] = Field(None, ge=0, le=1, description="Devierea ca procent (ex: 0.27 pentru 27%)")
    prediction: Optional[List[float]] = Field(None, description="Predicția AI pentru următoarele ore, ca listă de valori")
    anomaly_score: Optional[float] = Field(None, ge=0, le=1, description="Scorul de anomalie detectat de modelul autoencoder")
    shap_top_feature: Optional[str] = Field(None, description="Cel mai influent feature conform SHAP")
    shap_value: Optional[float] = Field(None, description="Valoarea SHAP a feature-ului cu impact maxim")

@router.post("/explanation")
async def explain_reason(data: ExplanationRequest):
    reasons = []
    recommendations = []

    # Explică devierea consumului/curentului
    if data.deviation is not None:
        if data.deviation > 0.2:
            reasons.append(f"Curentul a deviat cu {round(data.deviation * 100)}% față de trendul istoric, sugerând o schimbare semnificativă în consum.")
            recommendations.append("Monitorizează consumul pentru a preveni supraîncărcări.")
        else:
            reasons.append("Curentul este în limite normale față de trendul istoric.")

    # Explică temperatura și impactul ei
    if data.temperature > 40:
        reasons.append("Temperatura foarte ridicată depășește pragul de siguranță, risc de supraîncălzire.")
        recommendations.append("Verifică sistemul de răcire și limitează încărcarea dacă este necesar.")
    elif data.temperature > 35:
        reasons.append("Temperatura ridicată poate afecta performanța și siguranța stației.")
        recommendations.append("Asigură ventilația și evită încărcările intensive.")
    elif data.temperature < 0:
        reasons.append("Temperatura scăzută poate reduce eficiența sistemului de încărcare.")
        recommendations.append("Protejează stația împotriva frigului pentru maximizarea performanței.")
    else:
        reasons.append("Temperatura este în limite normale.")

    # Explică nivelul bateriei
    if data.battery_soc < 20:
        reasons.append("Nivelul scăzut al bateriei indică necesitatea începerii încărcării.")
        recommendations.append("Pornește încărcarea cât mai curând pentru a evita epuizarea bateriei.")
    elif data.battery_soc > 80:
        reasons.append("Nivelul ridicat al bateriei sugerează că încărcarea poate fi amânată.")
        recommendations.append("Programează încărcarea la ore cu tarife mai mici pentru eficiență.")
    else:
        reasons.append("Nivelul bateriei este optim pentru încărcare.")

    # Reguli avansate suplimentare bazate pe combinații de parametri
    if data.temperature is not None and data.current is not None:
        if data.temperature > 40 and data.current > 100:
            reasons.append("Există un risc ridicat de supraîncălzire datorită temperaturii și curentului simultan ridicate.")
            recommendations.append("Reduce setările de încărcare și monitorizează echipamentul imediat.")

    # Explică predicția LSTM AI (exemplu simplu pe baza valorilor maxime)
    if data.prediction:
        max_pred = max(data.prediction)
        if max_pred > 0.6:
            reasons.append(f"AI-ul prezice un vârf de consum semnificativ (+{max_pred:.2f}) în următoarele ore.")
            recommendations.append("Planifică încărcarea înainte de acest vârf pentru eficiență și economii.")
        else:
            reasons.append("Previziunea AI indică un consum stabil fără creșteri majore.")

    # Explică scorul de anomalie
    if data.anomaly_score is not None:
        if data.anomaly_score > 0.9:
            reasons.append(f"AI-ul a detectat o anomalie severă în funcționarea stației (scor: {data.anomaly_score:.2f}).")
            recommendations.append("Verifică și efectuează mentenanța preventivă a stației.")
        else:
            reasons.append("Nu s-au detectat anomalii recente în funcționarea stației.")

    # Explică importanța unui feature cu SHAP
    if data.shap_top_feature and data.shap_value is not None:
        reasons.append(f"Caracteristica '{data.shap_top_feature}' a avut cel mai mare impact asupra deciziei AI (valoare SHAP: {data.shap_value:.2f}).")
        recommendations.append(f"Monitorizează '{data.shap_top_feature}' pentru optimizarea performanței stației.")

    # Fallback dacă nu s-au găsit motive explicite
    if not reasons:
        reasons.append("Date insuficiente pentru a genera o explicație detaliată.")

    explanation_parts = ["Cauzele recomandării AI:"]
    explanation_parts.extend(f"- {r}" for r in reasons)

    if recommendations:
        explanation_parts.append("\nRecomandări AI:")
        explanation_parts.extend(f"- {r}" for r in recommendations)

    explanation_text = "\n".join(explanation_parts)

    return {"explanation": explanation_text}
