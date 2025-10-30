import React, { useState, useEffect } from "react";
import "./App.css";
import { LineChart, Line } from "recharts";
import { toast as notify, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import BookingForm from "./BookingForm";
import AuthForm from "./AuthForm";

// API endpoints, update PORT if needed
const API_URL = "http://localhost:8000/stations/full_status";
const ALERTS_URL = "http://localhost:8000/maintenance/alerts";
const PREDICT_URL = "http://localhost:8000/stations/predict_lstm";
const DYNAMIC_PRICE_URL = "http://localhost:8000/dynamic_price";
const RECOMMEND_URL = "http://localhost:8000/stations/recommend_smart_schedule";


// Metric definitions for the dashboard
const metricMeta = [
  { label: "Tensiune", key: "voltage", unit: "V", icon: "icon-voltage", tooltip: "Tensiunea livrată de stație" },
  { label: "Curent", key: "current", unit: "A", icon: "icon-current", tooltip: "Curentul consumat" },
  { label: "Temperatură", key: "temperature", unit: "°C", icon: "icon-temp", tooltip: "Temperatura sistemului" },
  { label: "Baterie", key: "battery_soc", unit: "%", icon: "icon-battery", tooltip: "Nivelul de încărcare al bateriei" }
];

function getStatusLedColor(status) {
  if (status === "online") return "green";
  if (status === "offline") return "red";
  return "yellow";
}

function App() {
  const [dark, setDark] = useState(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const [stations, setStations] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [aiForecast, setAiForecast] = useState({});
  const [aiPrice, setAiPrice] = useState({});
  const [recommend, setRecommend] = useState({});
  const [toast, setToast] = useState("");
  const [showQuick, setShowQuick] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [showExplain, setShowExplain] = useState(false);
  const [explainText, setExplainText] = useState("");
  const [user, setUser] = useState(null); // sau din autentificare reală
  const [showRegister, setShowRegister] = useState(false);

  const handleAuth = (authData) => {
    if (authData) {
      setUser(authData);
      localStorage.setItem("authToken", authData.token);
      localStorage.setItem("username", authData.username);
    } else {
      setUser(null);
      localStorage.removeItem("authToken");
      localStorage.removeItem("username");
    }
  };

  // Fetch stations and alert data from backend
  async function fetchData() {
    setLoading(true);
    try {
      const [stationsResp, alertsResp] = await Promise.all([
        fetch(API_URL),
        fetch(ALERTS_URL)
      ]);
      const stationsData = await stationsResp.json();
      const alertsData = await alertsResp.json();
      setStations(Array.isArray(stationsData) ? stationsData : stationsData.stations || []);
      setAlerts(Array.isArray(alertsData.alerts) ? alertsData.alerts : alertsData || []);
    } catch (error) {
      setToast("Eroare la preluarea datelor backend!");
    }
    setLoading(false);
  }

  useEffect(() => {
  // La montare, verifică dacă există token salvat în localStorage
  const token = localStorage.getItem("authToken");
  const username = localStorage.getItem("username");
  if(token && username) {
  setUser({ username: username, token: token });
  }
}, []);


  useEffect(() => {
    document.body.className = dark ? "dark-mode" : "";
  }, [dark]);

  useEffect(() => {
  if (user) {
    fetchData();
  } else {
    setStations([]); // sau poți să îl elimini dacă nu vrei să golești lista la logout
  }
}, [user]);

  useEffect(() => {
    fetchData();
  }, []);

  // Helper to get alerts for a single station
  function getStationAlerts(stationId) {
    return alerts.filter(a => a.station_id === stationId);
  }

  // ADMIN actions
  async function addStation(stationId = "EV003", status = "online") {
    try {
      const res = await fetch("http://localhost:8000/stations/add_station", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ station_id: stationId, status })
      });
      await res.json();
      fetchData();
      setToast("Stație adăugată!");
    } catch {
      setToast("Eroare la adăugarea stației!");
    }
  }

  async function deleteStation(stationId) {
    try {
      const res = await fetch("http://localhost:8000/stations/delete_station", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ station_id: stationId })
      });
      await res.json();
      fetchData();
      setToast("Stație ștearsă!");
    } catch {
      setToast("Eroare la ștergerea stației!");
    }
  }

  async function resetAllStations(status = "online") {
    try {
      await fetch("http://localhost:8000/stations/reset_all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status })
      });
      fetchData();
      setToast("Reset " + status);
    } catch {
      setToast("Eroare la resetare!");
    }
  }

  async function fetchStats() {
    try {
      const res = await fetch("http://localhost:8000/stations/stats");
      const data = await res.json();
      fetchData();
      setToast(`Total: ${data.total}, Online: ${data.online}, Offline: ${data.offline}`);
    } catch {
      setToast("Eroare la preluarea statistici!");
    }
  }

  async function downloadFullCSV() {
    try {
      const res = await fetch("http://localhost:8000/stations/export_csv");
      const text = await res.text();
      const blob = new Blob([text], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = "statii_ev_full.csv";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setToast("Export CSV complet realizat!");
    } catch {
      setToast("Eroare export CSV!");
    }
  }


  // AI Actions — by station
const handleAIPredict = async (station) => {
  // 1. Nu face predicție pentru stațiile offline
  if (station.status !== "online") {
    setAiForecast((prev) => ({
      ...prev,
      [station.station_id]: "Stația este offline, previziune indisponibilă."
    }));
     console.log(`Stația ${station.station_id} este offline, nu se face predicție.`);
    return;
  }

  try {
   const historyResp = await fetch(`http://localhost:8000/history/stations/${station.station_id}`);
if (!historyResp.ok) {
  throw new Error("Eroare la preluarea istoricului!");
}

const historyData = await historyResp.json();
console.log("historyData raw:", historyData);

 console.log(`Istoric pentru stația ${station.station_id}:`, historyData);

    // 3. Trimite datele istorice la backend pentru predicție AI
const predictResp = await fetch(`http://localhost:8000/stations/predict_lstm`, {

      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(historyData)
 // sau ajustează structura dacă îi trebuie doar valorile
    });
    if (!predictResp.ok) {
      throw new Error("Eroare predicție AI!");
    }
    const predictData = await predictResp.json();

    const predValues = predictData.prediction || []; // vector de valori numerice

    // 4. Generează mesaj adaptat pentru această stație
    if (predValues.length === 0) {
      setAiForecast((prev) => ({
        ...prev,
        [station.station_id]: "Nu există suficiente date pentru a oferi o previziune."
      }));
      return;
    }

    const maxVal = Math.max(...predValues);
    const minVal = Math.min(...predValues);
    const avgVal = predValues.reduce((a, b) => a + b, 0) / predValues.length;

    let message = "";

    if (maxVal > 1.0) {
      message += "- Atenție: se așteaptă un vârf foarte mare de consum.\n";
    } else if (maxVal > 0.5) {
      message += "- Vârf moderat de consum.\n";
    } else if (maxVal > 0.1) {
      message += "- Consum moderat așteptat.\n";
    } else {
      message += "- Consum/încărcare stabilă, fără vârfuri majore.\n";
    }

    if (minVal < -0.3) {
      message += "- Interval cu consum foarte scăzut, oportun pentru mentenanță.\n";
    } else if (minVal < 0) {
      message += "- Consum redus în anumite intervale.\n";
    }

    message += `- Valoarea medie estimată: ${avgVal.toFixed(2)} (unități specifice).`;

    // 5. Actualizare stare cu mesajul generat pentru această stație
    setAiForecast((prev) => ({
      ...prev,
      [station.station_id]: message
    }));

  } catch (error) {
    console.error(`Eroare la predicția AI pentru stația ${station.station_id}:`, error);
    setAiForecast((prev) => ({
      ...prev,
      [station.station_id]: "Eroare la generarea previziunii AI."
    }));
  }
};




const handleDynamicPrice = async (station) => {
  setAiPrice(prev => ({ ...prev, [station.station_id]: "Se calculează tarif AI..." }));
  try {
    // Preia datele istorice pentru stație
    const historyResp = await fetch(`http://localhost:8000/history/stations/${station.station_id}`);
    if (!historyResp.ok) throw new Error(`Eroare la încărcarea istoricului: ${historyResp.statusText}`);

    const historyData = await historyResp.json();
console.log("historyData raw:", historyData);

    // Normalizează array-ul de istoric
    const historyArray = Array.isArray(historyData) ? historyData : historyData.data || historyData.history || [];

    // Validare structură array 2D 24x8 numerice
    if (
      !Array.isArray(historyArray) ||
      historyArray.length !== 24 ||
      !Array.isArray(historyArray[0]) ||
      historyArray[0].length !== 8 ||
      !historyArray.every(row => row.every(val => typeof val === "number"))
    ) {
      throw new Error("Date istorice insuficiente sau format invalid pentru calcul tarif AI.");
    }

    console.log("Payload trimis la backend pentru tarif AI:", { payload: historyArray });
    // Trimite request POST cu CHEIA "payload" OBLIGATORIE, conform backend-ului
    const res = await fetch(DYNAMIC_PRICE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ payload: historyArray })  // aici cheia "payload" e necesară!
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Eroare la calcul tarif AI: ${res.status} - ${errorText}`);
    }

    const data = await res.json();

    let displayText = `${data.dynamic_price} €/kWh\n`;
    if (data.explanation) displayText += `Info: ${data.explanation}\n`;
    if (data.recommendation) displayText += `Recomandare: ${data.recommendation}\n`;
    if (data.valid_for) displayText += `Valabil pentru: ${data.valid_for}`;

    setAiPrice(prev => ({ ...prev, [station.station_id]: displayText }));
    setToast("Tarif AI calculat!");
  } catch (error) {
    console.error(error);
    setAiPrice(prev => ({ ...prev, [station.station_id]: `Eroare: ${error.message}` }));
    setToast("Eroare la calcul tarif AI!");
  }
};






  // Recomandare AI smart (aplicată general)
  const handleRecommend = async () => {
    setRecommend("...");
    try {
      const res = await fetch(RECOMMEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });
      const data = await res.json();
      setRecommend(data.suggestion || JSON.stringify(data));
      setToast("Recomandare AI generată!");
    } catch {
      setRecommend("Eroare la recomandare.");
    }
  };

const handleExplain = async (station) => {
  setExplainText("Se încarcă explicația...");
  setShowExplain(true);

  // Construiește payload cu valori sigure (nu undefined sau null), convertite la numere
  const bodyPayload = {
    temperature: (station.temperature !== undefined && station.temperature !== null) ? Number(station.temperature) : 0,
    current: (station.current !== undefined && station.current !== null) ? Number(station.current) : 0,
    battery_soc: (station.battery_soc !== undefined && station.battery_soc !== null) ? Number(station.battery_soc) : 0,
    deviation: (station.deviation !== undefined && station.deviation !== null) ? Number(station.deviation) : 0,
    // poți adăuga aici și alte câmpuri opționale dacă vrei
  };

  console.log("Payload explicație trimis:", bodyPayload);

  // Validare simplă înainte de trimitere
  if (
    typeof bodyPayload.temperature !== "number" ||
    bodyPayload.temperature < -50 || bodyPayload.temperature > 100 ||
    typeof bodyPayload.current !== "number" ||
    bodyPayload.current < 0 ||
    typeof bodyPayload.battery_soc !== "number" ||
    bodyPayload.battery_soc < 0 || bodyPayload.battery_soc > 100
  ) {
    console.error("Payload invalid - câmpuri obligatorii invalid setate", bodyPayload);
    setExplainText("Eroare: Date invalide pentru explicații AI.");
    return;
  }

  try {
    const res = await fetch("http://localhost:8000/explanation/explanation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bodyPayload),
    });

    if (!res.ok) {
      throw new Error(`Eroare la încărcarea explicației: ${res.statusText}`);
    }

    const data = await res.json();
    setExplainText(data.explanation || "Nu s-a primit explicație de la server.");
  } catch (e) {
    setExplainText("Eroare la încărcarea explicației.");
    console.error(e);
  }
};




  useEffect(() => {
    if (toast && toast.length) {
      const to = setTimeout(() => setToast(""), 2300);
      return () => clearTimeout(to);
    }
  }, [toast]);

  // Export CSV pentru toate stațiile din listă
  function exportCSV() {
    if (!stations.length) return;
    const csv = [
      ["ID", "Tensiune", "Curent", "Temperatură", "Baterie"],
      ...stations.map(s => [
        s.station_id, s.voltage, s.current, s.temperature, s.battery_soc
      ])
    ].map(row => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "statii_ev.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setToast("Export CSV rapid realizat!");
  }

  useEffect(() => {
    function handleKey(e) {
      if (e.ctrlKey && e.code === "KeyE") {
        e.preventDefault();
        exportCSV();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [stations]);

  function notifyAlert(alert) {
    notify.info(`${alert.station_id}: ${alert.alert}`);
  }

  const leaderboard = [
    { name: "Carina", eco: "Super Eco", ecoValue: 93 },
    { name: "Mihai", eco: "Eco", ecoValue: 71 },
    { name: "Alex", eco: "Basic", ecoValue: 52 }
  ];

 return (
  <div className={`App${dark ? " dark-mode" : ""}`}>
    {/* Theme Switch */}
    <div className="theme-switch">
      <input
        type="checkbox"
        checked={dark}
        onChange={() => setDark(d => !d)}
        id="themeCheck"
      />
      <label htmlFor="themeCheck" style={{ fontSize: 14, color: dark ? "#fff" : "#2196f3" }}>
        Dark mode
      </label>
    </div>

    <h1 style={{ marginTop: 0 }}>Stații Încărcare SmartCharge AI</h1>
    
     {!user ? (
      // Dacă NU este autentificat, afișăm formularul AuthForm și butonul pentru switch Login/Register
      <>
        <AuthForm
          type={showRegister ? "register" : "login"}
          onAuth={handleAuth}
        />
        <button
          style={{
            display: "block",
            margin: "12px auto",
            background: "none",
            border: "none",
            color: "#2196f3",
            cursor: "pointer",
            textDecoration: "underline",
            fontSize: 15,
          }}
          onClick={() => setShowRegister(r => !r)}
          aria-label="Schimbă între Login și Înregistrare"
        >
          {showRegister ? "Ai deja cont? Autentifică-te" : "Nu ai cont? Creează unul"}
        </button>
      </>
    ) : (
      // Dacă este autentificat, afișăm dashboard-ul complet
      <>
        {/* Bine ai venit + buton logout */}
        <div style={{ marginBottom: 15, display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap" }}>
          <span style={{ color: "#2196f3", fontWeight: "bold" }}>
            Bun venit, {user.username}!
          </span>
          <button
            onClick={() => handleAuth(null)}
            style={{
              padding: "6px 14px",
              fontSize: 14,
              cursor: "pointer",
              backgroundColor: "#e53935",
              color: "#fff",
              border: "none",
              borderRadius: 6
            }}
            aria-label="Deloghează-te"
          >
            Deloghează-te
          </button>
        </div>
      
        

        {/* Loader */}
        <img className="loader-gif" src="/gifs/charging_ev.gif" alt="Se încarcă..." style={{ display: loading ? "block" : "none" }} />

        {/* Mesaj când nu sunt stații */}
        {!loading && stations.length === 0 && (
          <div className="dashboard-card" style={{ margin: 16, fontWeight: 600, color: "#ef5350" }}>
            Nicio stație de încărcare disponibilă!
          </div>
        )}

        {/* Lista stațiilor */}
        {!loading && stations.length > 0 && stations.map((s) => (
          <div className="dashboard-card" key={s.station_id} style={{ marginBottom: 32 }}>
            {/* Detalii stație */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6, justifyContent: "center" }}>
              <span className={`status-led ${getStatusLedColor(s.status)}`} title={s.status}></span>
              <span style={{ fontWeight: 600, fontSize: 16, color: dark ? "#fff" : "#2196f3" }}>
                {s.station_id}
              </span>
              <span className="eco-badge" style={{ animation: "eco-bounce 1s infinite alternate" }}>
                Eco+
              </span>
              {s.status !== "online" && <span className="badge-new">offline</span>}
            </div>

            <div className="metrics">
              {metricMeta.map((m) => (
                <div className="metric-item" key={m.key}>
                  <span className="metric-label tooltip">
                    <span className={m.icon}></span>
                    {m.label}
                    <span className="tooltiptext">{m.tooltip}</span>
                  </span>
                  <span className="metric-value">
                    {s[m.key] !== undefined && s[m.key] !== null ? `${s[m.key]}${m.unit}` : "-"}
                  </span>
                  {m.key !== "battery_soc" && (
                    <span className="sparkline">
                      <LineChart width={88} height={25} data={s.history || []}>
                        <Line type="monotone" dataKey={m.key} stroke="#2196f3" dot={false} strokeWidth={1.7} />
                      </LineChart>
                    </span>
                  )}
                </div>
              ))}
            </div>

            {s.battery_soc != null && (
              <>
                <div className="battery-bar" title="Nivel baterie">
                  <div className="battery-level" style={{ width: `${s.battery_soc}%` }} />
                </div>
                <div style={{ margin: "22px auto 0 auto", position: "relative", width: "54px" }}>
                  <div className="circular-progress" aria-label="Progress nivel baterie" />
                  <div className="circular-progress-text">{Math.round(s.battery_soc)}%</div>
                </div>
              </>
            )}

            {/* AI Buttons & Results */}
            <div style={{ marginTop: 18, display: "flex", flexWrap: "wrap", gap: 12, justifyContent: "center" }}>
              <button className="btn-primary" onClick={() => handleAIPredict(s)} style={{ fontSize: 15 }}>
                Previziune AI
              </button>
              <button className="btn-primary" onClick={() => { console.log("Clicked Tarif AI pentru stație", s.station_id); handleDynamicPrice(s); }} style={{ fontSize: 15 }}>
                  Tarif AI
              </button>

              <button
                style={{ fontSize: 14, background: "transparent", color: dark ? "#30b7f5" : "#2196f3", border: "none", cursor: "pointer", paddingLeft: 0 }}
                onClick={() => handleExplain(s)}
              >
                De ce această recomandare?
              </button>
            </div>

            {/* AI forecast */}
            {aiForecast[s.station_id] && (
              typeof aiForecast[s.station_id] === "string" ? (
                <div style={{ marginTop: 13, color: "#2196f3", whiteSpace: "pre-line" }}>
                  {aiForecast[s.station_id]}
                </div>
              ) : (
                <div style={{ marginTop: 12, color: "#2196f3" }}>
                  <b>Forecast (multi-step):</b>
                  <ul style={{
                    fontSize: 12,
                    background: "#e3eafc",
                    borderRadius: 9,
                    padding: 8,
                    marginTop: 6,
                    color: "#202052",
                    maxHeight: 180,
                    overflowY: 'auto',
                    listStyleType: "decimal",
                    textAlign: "left",
                  }}>
                    {aiForecast[s.station_id].map((val, idx) => {
                      if (Array.isArray(val)) {
                        return <li key={idx}>{val.map(v => v.toFixed(3)).join(", ")}</li>;
                      }
                      return <li key={idx}>{val.toFixed(3)}</li>;
                    })}
                  </ul>
                </div>
              )
            )}

            {/* AI price */}
            {aiPrice[s.station_id] && (
              typeof aiPrice[s.station_id] === "string" ? (
                <div style={{ marginTop: 16, color: "#388e3c" }}>
                  <b>Tarif dinamic AI:</b> {aiPrice[s.station_id]}
                </div>
              ) : (
                <div style={{ marginTop: 16, color: "#388e3c" }}>
                  <b>Tarif dinamic AI:</b> {JSON.stringify(aiPrice[s.station_id], null, 2)}
                </div>
              )
            )}

            {/* Recomandare */}
            {recommend && typeof recommend === "string" && (
              <div style={{ marginTop: 17 }}>
                <span className="eco-badge" style={{ animation: "eco-bounce 1.6s infinite alternate" }}>
                  +5 ECO
                </span>
                <b style={{ color: "#2196f3", marginLeft: 7 }}>{recommend}</b>
              </div>
            )}

            {getStationAlerts(s.station_id).length > 0 && (
              <div style={{ marginTop: 19 }}>
                {getStationAlerts(s.station_id).map((a, idx) =>
                  <div key={idx} style={{ color: "#e040fb", fontWeight: 600, fontSize: 15, margin: "6px 0" }}>
                    <span className="status-led red" style={{ marginRight: 7 }} />
                    {a.alert} <span style={{ fontSize: 12, color: "#888", marginLeft: 5 }}>{a.timestamp && ("" + a.timestamp).slice(0, 16)}</span>
                  </div>
                )}
              </div>
            )}

            <button
              className="btn-primary"
              style={{ background: '#e53935', marginTop: 12 }}
              onClick={() => deleteStation(s.station_id)}
              aria-label={`Șterge stația ${s.station_id}`}
              title={`Șterge stația ${s.station_id}`}
            >
              Șterge această stație
            </button>
          </div>
        ))}

        {/* BookingForm */}
        <BookingForm
          user={user.username}   // PASĂ DOAR username-ul
          stations={stations}
          onBookingSuccess={() => {
            setToast("Rezervare făcută cu succes");
            fetchData();
          }}
        />
      </>
    )}

    {/* Leaderboard Eco */}
    <div className="leaderboard" aria-label="Top Eco Utilizatori">
      <h3 style={{ color: "#2196f3", margin: 0, fontSize: "1.07rem" }}>Top Eco Utilizatori</h3>
      {leaderboard.map((u, idx) => (
        <div key={idx} style={{ marginTop: 8, marginBottom: 7, display: "flex", alignItems: "center" }}>
          <b style={{ fontSize: 19, color: "#b5b5b5", marginRight: 8 }}>{idx + 1}</b>
          <span style={{ fontWeight: 600, color: "#23262f" }}>{u.name}</span>
          <span className="eco-badge" style={{ marginLeft: 9 }}>{u.eco}</span>
          <div className="progress-objective" style={{ width: 60, marginLeft: 8 }}>
            <div className="progress-bar" style={{ width: `${u.ecoValue}%` }}></div>
          </div>
        </div>
      ))}
    </div>

    {/* Notifications feed */}
    <div className="notifications-feed" aria-live="polite" aria-atomic="true">
      {alerts.map((a, i) => (
        <div className="notif-entry" key={a.station_id + i} onClick={() => notifyAlert(a)} tabIndex={0} role="button" aria-label={`Alertă pentru stația ${a.station_id}: ${a.alert}`}>
          <span className={`status-led ${getStatusLedColor(a.status || "offline")}`}></span>
          <span>
            {a.station_id}: {a.alert}
            <span className="badge-new" style={{ marginLeft: 7 }}>Atenție</span>
          </span>
        </div>
      ))}
    </div>

    {/*
  Quick Settings dezvoltat cu butoane funcționale, înlocuind conținutul gol
*/}
<>
  {showQuick && (
    <div
      className="quick-settings"
      role="region"
      aria-label="Setări rapide"
      style={{
        position: "fixed",
        bottom: 80,
        right: 28,
        zIndex: 9999,
        backgroundColor: "rgba(255, 255, 255, 0.98)",
        padding: 22,
        borderRadius: 16,
        minWidth: 280,
        maxWidth: 320,
        boxShadow: "0 4px 12px rgba(0,0,0,0.18)",
      }}
    >
      <button
        className="btn-primary"
        style={{ width: "100%", marginBottom: 13 }}
        onClick={() => setShowQuick(false)}
      >
        Închide setări rapide
      </button>

      <button
        className="btn-primary"
        style={{ width: "100%", marginBottom: 11 }}
        onClick={() => addStation()}
      >
        Adaugă stație nouă
      </button>

      <button
        className="btn-primary"
        style={{ width: "100%", marginBottom: 11 }}
        onClick={() => resetAllStations("online")}
      >
        Resetează toate la online
      </button>

      <button
        className="btn-primary"
        style={{ width: "100%", marginBottom: 11 }}
        onClick={() => resetAllStations("offline")}
      >
        Resetează toate la offline
      </button>

      <button
        className="btn-primary"
        style={{ width: "100%", marginBottom: 13 }}
        onClick={handleRecommend}
      >
        Recomandare optimă AI
      </button>
      <button className="btn-primary" style={{ marginTop: 8, fontSize: 15 }} onClick={exportCSV}>
          Export date (CSV)
        </button>
    </div>
  )}

  {/* Butonul floating de deschidere toggle */}
  <button
    className="btn-primary"
    style={{
      position: "fixed",
      bottom: 38,
      right: 28,
      zIndex: 9999,
      fontSize: 16,
      background: "#43a047",
    }}
    onClick={() => setShowQuick(prev => !prev)}
    aria-label="Deschide/Inchide setări rapide"
  >
    Setări rapide
  </button>
</>




    {/* Onboarding overlay */}
    {showOnboarding && (
      <div className="onboarding-overlay" onClick={() => setShowOnboarding(false)} role="dialog" aria-modal="true" aria-labelledby="onboarding-title" tabIndex={-1}>
        <div className="onboarding-card">
          <h2 id="onboarding-title">Bun venit în SmartCharge AI!</h2>
          <p>
            Dashboard-ul afișează <b>stațiile</b> și parametrii EV în timp real, cu funcții AI:<br />
            previziuni, tarife smart, recomandări proactive și alerte automate!
          </p>
          <button
            className="btn-primary"
            style={{ marginTop: 19 }}
            onClick={() => setShowOnboarding(false)}
            aria-label="Închide mesajul de bun venit"
          >
            Am înțeles
          </button>
        </div>
      </div>
    )}

    {/* Modal AI Explanation */}
    {showExplain && (
      <div
        style={{
          position: "fixed", left: 0, top: 0, right: 0, bottom: 0, zIndex: 99999,
          background: "rgba(33,150,243,0.12)", display: "flex", alignItems: "center", justifyContent: "center"
        }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="explain-title"
        tabIndex={-1}
        onClick={() => setShowExplain(false)}
      >
        <div
          style={{
            background: "#fff", borderRadius: 18, boxShadow: "0 8px 32px #2196f333", padding: 34, textAlign: "center",
            maxWidth: "90vw", maxHeight: "80vh", overflowY: "auto"
          }}
          onClick={e => e.stopPropagation()}
        >
          <h3 id="explain-title">Explicație AI</h3>
          <p style={{ color: "#2196f3", fontSize: 17 }}>{explainText}</p>
          <button className="btn-primary" onClick={() => setShowExplain(false)} aria-label="Închide explicația AI">OK</button>
        </div>
      </div>
    )}
  </div>
);
}

export default App;
