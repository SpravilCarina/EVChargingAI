from fastapi import APIRouter, Body, HTTPException
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import logging

router = APIRouter()

# Încarcă modelul LSTM pentru predicții
model = load_model(r"D:\ev-charging-ai\backend\app\ev_lstm_tfmodel.h5", compile=False)

history_steps = 24
feature_count = 8

logging.basicConfig(level=logging.INFO)

@router.post("/")
async def dynamic_price(body: dict = Body(...)):
    """
    Endpoint AI Dynamic Price:
    - Primește JSON cu cheia 'payload' care conține lista (24,8)
    - Procesează datele și returnează tarif AI și alte informații
    """
    if "payload" not in body:
        raise HTTPException(status_code=422, detail="Cheia 'payload' este obligatorie în corpul cererii.")

    payload = body["payload"]
    logging.info(f"[DynamicPrice] Payload primit de volum {len(payload)} și shape aproximativ: {np.array(payload).shape}")

    # Validare structură și tipuri
    if not isinstance(payload, list):
        raise HTTPException(status_code=422, detail="Payload-ul trebuie să fie o listă (matrice 24x8).")
    if len(payload) != history_steps:
        raise HTTPException(status_code=422, detail=f"Payload trebuie să aibă {history_steps} rânduri, dar are {len(payload)}.")
    for i, row in enumerate(payload):
        if not isinstance(row, list):
            raise HTTPException(status_code=422, detail=f"Rândul {i} din payload nu este o listă.")
        if len(row) != feature_count:
            raise HTTPException(status_code=422, detail=f"Rândul {i} trebuie să aibă {feature_count} elemente, are {len(row)}.")
        for j, val in enumerate(row):
            if not isinstance(val, (int, float)):
                raise HTTPException(status_code=422, detail=f"Valoarea de la poziția ({i},{j}) nu este numerică.")

    try:
        arr = np.array(payload)
        logging.info(f"[DynamicPrice] Payload array shape: {arr.shape}")

        arr_input = arr.reshape(1, history_steps, feature_count)
        pred = model.predict(arr_input).flatten()

        # Statistici forecast
        predicted_peak = float(np.max(pred))
        predicted_avg = float(np.mean(pred))
        predicted_std = float(np.std(pred))
        conf_int = (
            round(predicted_avg - 1.96 * predicted_std, 3),
            round(predicted_avg + 1.96 * predicted_std, 3)
        )

        # Trend consum
        trend = "UP" if pred[-1] > pred[0] + 0.05 else "DOWN" if pred[-1] < pred[0] - 0.05 else "STABIL"

        # Factor principal (exemplu)
        main_factor = "current"
        main_factor_importance = 0.15

        anomaly_score = float(np.std(arr))
        anomaly_flag = anomaly_score > 5.0

        price_spot = 0.42  # Exemplu valoare externă

        base_price = 1.0
        coef_peak = 0.5
        coef_avg = 0.1
        coef_spot = 0.6
        dynamic_price = base_price + coef_peak * predicted_peak + coef_avg * predicted_avg + coef_spot * price_spot
        dynamic_price = float(np.clip(dynamic_price, 0.80, 5.00))
        dynamic_price = round(dynamic_price, 3)

        explanation = []
        if anomaly_flag:
            explanation.append("Atenție: datele istorice sunt atipice, tarif AI stabilit cu prudență.")
        if predicted_peak > 1.0:
            explanation.append(f"Tariful este mai mare deoarece AI prevede vârf de consum {predicted_peak:.2f} în orele următoare.")
        elif predicted_avg < 0.2:
            explanation.append("Tariful e minim, AI-ul prezice consum scăzut.")
        else:
            explanation.append("Tariful reflectă estimarea AI a consumului viitor.")
        explanation.append(f"Tariful include corecție după prețul pieței: {price_spot:.2f} €/kWh.")

        if main_factor and main_factor_importance > 0.05:
            explanation.append(f"Factor major AI: '{main_factor}' (importanță: {main_factor_importance:.2f}).")

        recommendation = ""
        if dynamic_price > 2.0:
            recommendation = "Evită încărcarea în orele de vârf pentru a reduce costurile."
        elif dynamic_price < 1.0:
            recommendation = "Cost optim: încarcă fără restricții."
        elif trend == "UP":
            recommendation = "Se așteaptă creștere a cererii: planifică încărcarea cât mai devreme."

        now = datetime.now()
        logging.info(
            f"[DynamicPrice] {now}: Price: {dynamic_price}, Peak: {predicted_peak:.2f}, Avg: {predicted_avg:.2f}, "
            f"Std: {predicted_std:.2f}, Trend: {trend}, Spot: {price_spot}, Recom: {recommendation}"
        )

        return {
            "dynamic_price": dynamic_price,
            "ai_peak": round(predicted_peak, 3),
            "ai_avg": round(predicted_avg, 3),
            "ai_std": round(predicted_std, 3),
            "ai_forecast_sample": [float(v) for v in pred[:6]],
            "confidence_interval": conf_int,
            "main_factor": main_factor,
            "main_factor_importance": round(main_factor_importance, 2),
            "explanation": " ".join(explanation),
            "recommendation": recommendation,
            "trend": trend,
            "anomaly_score": round(anomaly_score, 3),
            "valid_for": f"{now.strftime('%Y-%m-%d %H:00')} - următoarele 6 ore",
            "price_spot": price_spot,
            "model_version": "LSTM_v1.0"
        }
    except Exception as e:
        logging.error(f"[DynamicPrice] ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Eroare la calcul tarif AI: {str(e)}")
