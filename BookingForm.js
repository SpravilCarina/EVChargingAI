import React, { useState, useEffect } from "react";

function BookingForm({ user, stations, onBookingSuccess }) {
  const [stationId, setStationId] = React.useState(stations?.length ? stations[0].station_id : "");
  const [slotStart, setSlotStart] = React.useState("");
  const [slotEnd, setSlotEnd] = React.useState("");
  const [error, setError] = React.useState(null);
  const [success, setSuccess] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [myBookings, setMyBookings] = React.useState([]);
  const [loadingBookings, setLoadingBookings] = React.useState(false);

  // Convert datetime-local string (local time) to ISO string with UTC 'Z'
  const toISOStringWithTimezone = (dateString) => {
    if (!dateString) return "";
    const dt = new Date(dateString);
    return dt.toISOString();
  };

  // Fetch current user's bookings
const fetchMyBookings = async () => {
  if (!user) return;
  setLoadingBookings(true);
  try {
    const username = typeof user === "string" ? user : user.username;
    console.log("Caut rezervări pentru user:", username);
    const res = await fetch(`http://localhost:8000/bookings/user/${encodeURIComponent(username)}`);
    if (!res.ok) throw new Error(`Eroare la preluarea rezervărilor: ${res.statusText}`);
    const data = await res.json();
    console.log("Răspuns rezervări:", data);
    setMyBookings(data || []);
    setError(null);
  } catch (err) {
    setError(err.message);
  } finally {
    setLoadingBookings(false);
  }
};


  // Load bookings when user changes
  useEffect(() => {
    fetchMyBookings();
  }, [user]);

  // Cancel booking (needs booking details to identify)
  const handleCancelBooking = async (booking) => {
    if (!window.confirm("Sigur doriți să anulați această rezervare?")) return;
    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const payload = {
  station_id: booking.station_id,
  user_id: booking.user_id, // sau booking.user dacă asta există exact
  slot_start: booking.slot_start,

};

      const res = await fetch("http://localhost:8000/bookings/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errResp = await res.json();
        throw new Error(errResp.detail || res.statusText);
      }
      setSuccess("Rezervare anulată cu succes.");
      await fetchMyBookings();
      if (typeof onBookingSuccess === "function") onBookingSuccess();
    } catch (err) {
      setError("Eroare la anularea rezervării: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Request AI recommendation for optimal booking slot
  const handleRecommendOptimalSlot = async () => {
    setError(null);
    setSuccess(null);
    if (!stationId) {
      setError("Selectați mai întâi o stație.");
      return;
    }

    setLoading(true);
    try {
      const historyResp = await fetch(`http://localhost:8000/history/stations/${stationId}`);
      if (!historyResp.ok) throw new Error("Eroare la preluarea istoricului stației.");
      const historyData = await historyResp.json();

      const predResp = await fetch(`http://localhost:8000/stations/predict_lstm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(historyData),

      });
      if (!predResp.ok) throw new Error("Eroare la predicția AI.");

      const predData = await predResp.json();
      const forecast = predData.prediction || [];
      if (!Array.isArray(forecast) || forecast.length !== 24)
        throw new Error("Previziune incorectă.");

      let minIndex = 0;
      let minValue = forecast[0];
      for (let i = 1; i < 24; i++) {
        if (forecast[i] < minValue) {
          minValue = forecast[i];
          minIndex = i;
        }
      }

      // Recommend slot starting at minIndex hour UTC (today)
      const now = new Date();
      const slotStartDateUtc = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), minIndex, 0, 0));
      const slotEndDateUtc = new Date(slotStartDateUtc.getTime() + 60 * 60 * 1000); // +1 hour slot

      // Convert to local datetime-local input format string
      const toLocalDateTimeString = (date) => {
        const tzOffset = date.getTimezoneOffset() * 60000;
        const localDate = new Date(date.getTime() - tzOffset);
        return localDate.toISOString().slice(0, 16);
      };

      setSlotStart(toLocalDateTimeString(slotStartDateUtc));
      setSlotEnd(toLocalDateTimeString(slotEndDateUtc));
      setSuccess(`AI recomandă intervalul ${slotStartDateUtc.getUTCHours()}:00 - ${slotEndDateUtc.getUTCHours()}:00 UTC.`);
    } catch (err) {
      setError(err.message || "Eroare la calcularea recomandării.");
    } finally {
      setLoading(false);
    }
  };
const normalizedUserId = typeof user === "string" ? user.toLowerCase() : (user.username ? user.username.toLowerCase() : "");

  // Submit booking form
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!slotStart || !slotEnd) {
      setError("Trebuie să selectați intervalul.");
      return;
    }
    if (new Date(slotStart) >= new Date(slotEnd)) {
      setError("Data începerii trebuie să fie înaintea celei de sfârșit.");
      return;
    }

    setLoading(true);
    try {
     const payload = {
  station_id: stationId,
  user_id: normalizedUserId,
  slot_start: toISOStringWithTimezone(slotStart),
  slot_end: toISOStringWithTimezone(slotEnd),
};

console.log("Rezervare payload trimis:", payload);

      const res = await fetch("http://localhost:8000/bookings/reserve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (res.ok) {
        setSuccess("Rezervare efectuată cu succes.");
        setSlotStart("");
        setSlotEnd("");
        await fetchMyBookings();
        if (onBookingSuccess) onBookingSuccess();
      } else {
        throw new Error(data.detail || "Eroare la rezervare.");
      }
    } catch (err) {
      setError("Eroare la rezervare: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 480, margin: "20px auto", padding: 12, border: "1px solid #ccc", borderRadius: 6 }}>
      <h3>Rezervă un slot la stație</h3>

      <form onSubmit={handleSubmit}>
        <label>
          Stație:
          <select
            value={stationId}
            onChange={(e) => setStationId(e.target.value)}
            required
            disabled={loading}
            style={{ marginLeft: 8, minWidth: 120 }}
          >
            {stations.map((station) => (
              <option key={station.station_id} value={station.station_id}>
                {station.station_id}
              </option>
            ))}
          </select>
        </label>

        <div style={{ marginTop: 12 }}>
          <button type="button" onClick={handleRecommendOptimalSlot} disabled={loading}>
            Recomandă ora optimă
          </button>
        </div>

        <div style={{ marginTop: 10 }}>
          <label>
            Început:
            <input
              type="datetime-local"
              required
              value={slotStart}
              onChange={(e) => setSlotStart(e.target.value)}
              disabled={loading}
              style={{ marginLeft: 8 }}
            />
          </label>
        </div>

        <div style={{ marginTop: 10 }}>
          <label>
            Sfârșit:
            <input
              type="datetime-local"
              required
              value={slotEnd}
              onChange={(e) => setSlotEnd(e.target.value)}
              disabled={loading}
              style={{ marginLeft: 8 }}
            />
          </label>
        </div>

        {error && <div style={{ color: "red", marginTop: 10, fontWeight: "bold" }}>{error}</div>}
        {success && <div style={{ color: "green", marginTop: 10, whiteSpace: "pre-wrap" }}>{success}</div>}

        <button type="submit" disabled={loading} style={{ marginTop: 15, padding: "6px 14px", fontSize: 14 }}>
          {loading ? "Se încarcă..." : "Rezervă"}
        </button>
      </form>

      <hr style={{ margin: "20px 0" }} />

      <h4>Rezervările mele</h4>

      {loadingBookings && <p>Se încarcă rezervările...</p>}
      {!loadingBookings && myBookings.length === 0 && <p>Nu aveți rezervări active.</p>}

      {!loadingBookings && myBookings.length > 0 && (
        <ul>
          {myBookings.map((b) => (
  <li key={b.id} style={{ marginBottom: 8 }}>
    <strong>{b.station_id}</strong>:
    {new Date(b.slot_start).toLocaleString()} - {new Date(b.slot_end).toLocaleString()}
    <br />
    <button
      type="button"
      disabled={loading}
      onClick={() => handleCancelBooking(b)}
      style={{ marginTop: 4 }}
    >
      Anulează rezervare
    </button>
  </li>
))}


        </ul>
      )}
    </div>
  );
}

export default BookingForm;
