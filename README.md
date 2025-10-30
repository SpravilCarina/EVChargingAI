EV Charging AI Management
Sistem inteligent de management al stațiilor de încărcare pentru vehicule electrice, bazat pe inteligență artificială. Proiectul integrează monitorizare în timp real, optimizare energetică, predicții de cerere și detecție de anomalii, oferind interfață web și API complet.

Funcționalități principale
Monitorizare și analiză în timp real (tensiune, curent, temperatură, stare baterie)

Predicția și programarea sesiunilor de încărcare

Optimizarea distribuției energiei între stații (load balancing)

Detecția automată a anomaliilor și mentenanță predictivă

Aplicație web pentru rezervare, plată și vizualizare date

Tehnologii utilizate
Backend: FastAPI (Python 3.x)

Machine Learning: scikit-learn, TensorFlow/PyTorch, pandas, numpy, SHAP

Bază de date: InfluxDB (time-series data)

Frontend: React.js

Vizualizare: matplotlib, seaborn, plotly

Containerizare: Docker Compose

Testare API: Postman

Structură proiect
text
ev-charging-ai/
│
├── backend/
│   ├── main.py
│   ├── models/
│   ├── routes/
│   └── ml/
│
├── frontend/
│   ├── src/
│   └── public/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docker-compose.yml
└── README.md
Instrucțiuni de instalare
Clonează depozitul:

bash
git clone https://github.com/Carina/ev-charging-ai.git
cd ev-charging-ai
Pornește baza de date InfluxDB:

bash
./influxd
Pornește serverul backend (FastAPI):

bash
uvicorn main:app --reload
Pornește interfața web:

bash
cd frontend
npm install
npm start
Accesează aplicația pe:
http://localhost:3000 pentru frontend
http://localhost:8000/docs pentru API

Direcții viitoare
Integrare OCPP 2.1 și digital twin

Implementare federated learning

Raportare ESG automată și certificare ISO 27001

Autor
Carina – Proiect Universitatea Politehnica Timișoara, 2025.
