from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# --- MONITORIZARE ---

class StationStatus(BaseModel):
    station_id: str = Field(..., example="EV001")
    status: str = Field(..., example="online")  # "online", "offline"
    voltage: Optional[float] = Field(None, example=230.2)
    current: Optional[float] = Field(None, example=16.1)
    temperature: Optional[float] = Field(None, example=37.3)
    battery_soc: Optional[float] = Field(None, example=54.1)  # State-of-Charge (%)
    timestamp: Optional[datetime] = Field(None, example="2024-03-12T09:05:00")


class StationsStatusResponse(BaseModel):
    stations: List[StationStatus]

# --- REZERVĂRI ---

class BookingRequest(BaseModel):
    station_id: str = Field(..., example="EV001")
    user_id: str = Field(..., example="USR123")
    slot_start: datetime = Field(..., example="2024-03-12T10:30:00")
    slot_end: datetime = Field(..., example="2024-03-12T12:00:00")

class BookingResponse(BaseModel):
    status: str = Field(..., example="success")
    booking_id: Optional[str] = Field(None, example="BKG567")
    details: Optional[dict] = None

class BookingHistoryItem(BaseModel):
    booking_id: str
    station_id: str
    user_id: str
    slot_start: datetime
    slot_end: datetime
    status: str

class BookingHistoryResponse(BaseModel):
    bookings: List[BookingHistoryItem]

# --- NOTIFICĂRI/MENTENANȚĂ ---

class MaintenanceAlert(BaseModel):
    station_id: str = Field(..., example="EV001")
    alert: str = Field(..., example="Temperatură ridicată")
    severity: Optional[str] = Field(None, example="critical") # "info", "warning", "critical"
    timestamp: datetime = Field(..., example="2024-03-12T09:17:00")

class MaintenanceAlertsResponse(BaseModel):
    alerts: List[MaintenanceAlert]

# --- ISTORIC ȘI PREVIZIUNI ---

class StationUsageRecord(BaseModel):
    timestamp: datetime
    energy_kwh: float
    session_duration_min: int
    user_id: Optional[str]

class UsageHistoryResponse(BaseModel):
    station_id: str
    history: List[StationUsageRecord]

class UsagePredictionResponse(BaseModel):
    station_id: str
    predicted_usage_kwh: float
    prediction_timestamp: datetime

# --- UTILIZATOR (opțional, pentru autentificare sau detalii user) ---

class UserModel(BaseModel):
    user_id: str
    username: str
    email: Optional[str]
    is_active: Optional[bool] = True

# --- ERROR RESPONSE STANDARD ---

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
