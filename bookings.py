from fastapi import APIRouter, HTTPException, Query
from fastapi import Body
from pydantic import BaseModel
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from typing import List, Optional
import os


router = APIRouter()


# === Load secrets from env variables for security ===
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "WDo6QMstKcq2TvrA355ST8B7wowGyDEafy2R_C1rZmd_WqB5nRrYYCsFLmDt-NnT0URHhGGx1mUcYJNvJDsgHQ==")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "ev-charging-ai")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "incarcare_ev")


client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()


# === Pydantic Schemas ===

class BookingRequest(BaseModel):
    station_id: str
    user_id: str
    slot_start: datetime
    slot_end: datetime


class BookingResponse(BaseModel):
    id: str  # Unique ID: combination of user+station+slot_start timestamp
    station_id: str
    user_id: str
    slot_start: datetime
    slot_end: datetime
    cancelled: bool = False  # default False

class CancelRequest(BaseModel):
    station_id: str
    user_id: str
    slot_start: datetime
# === Helper functions ===

def slots_overlap(start1: float, end1: float, start2: float, end2: float) -> bool:
    """Return True if time intervals [start1,end1) and [start2,end2) overlap."""
    return start1 < end2 and start2 < end1


def make_booking_id(station_id: str, user_id: str, slot_start_ts: float) -> str:
    """Generate unique booking ID."""
    return f"{user_id}_{station_id}_{int(slot_start_ts)}"


# === Routes ===

@router.post("/reserve", response_model=BookingResponse)
async def reserve_session(request: BookingRequest):
    # Validate interval
    if request.slot_start >= request.slot_end:
        raise HTTPException(status_code=400, detail="Slot start must be before slot end.")
    if request.slot_start < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Cannot create bookings in the past.")
    duration_hours = (request.slot_end - request.slot_start).total_seconds() / 3600
    if duration_hours > 24:
        raise HTTPException(status_code=400, detail="Maximum booking duration is 24 hours.")

    start_ts = request.slot_start.timestamp()
    end_ts = request.slot_end.timestamp()

    # Query active (not cancelled) bookings overlapping with requested slot on same station

    query = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
          |> range(start: 0)
          |> filter(fn: (r) => r["_measurement"] == "booking")
          |> filter(fn: (r) => r["station_id"] == "{request.station_id}")
          |> filter(fn: (r) => r["_field"] == "slot_start" or r["_field"] == "slot_end" or r["_field"] == "cancelled")
          |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
          |> filter(fn: (r) => exists r.slot_start and exists r.slot_end and exists r.cancelled)
          |> filter(fn: (r) => r.cancelled == false)



    '''
    try:
        tables = query_api.query(query, org=INFLUXDB_ORG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying bookings: {str(e)}")

    for table in tables:
        points_by_time = {}
        # Group points by _time (to get both slot_start and slot_end fields)
        for record in table.records:
            time_key = record.get_time()
    # Sari peste rânduri fără '_value'
            if '_value' not in record.values:
                continue
            points_by_time.setdefault(time_key, {})
            points_by_time[time_key][record.get_field()] = record.get_value()
            points_by_time[time_key]["station_id"] = record.values.get("station_id")
            points_by_time[time_key]["user_id"] = record.values.get("user_id")
        # Check overlap per grouped point
        for time_key, fields in points_by_time.items():
            try:
                booked_start = float(fields.get("slot_start", 0))
                booked_end = float(fields.get("slot_end", 0))
                
                # Ignore cancelled bookings with slot_end == 0
                if booked_end == 0:
                    continue

                if slots_overlap(start_ts, end_ts, booked_start, booked_end):
                    booked_start_dt = datetime.utcfromtimestamp(booked_start)
                    booked_end_dt = datetime.utcfromtimestamp(booked_end)
                    raise HTTPException(
                        status_code=409,
                        detail=f"The requested slot overlaps with an existing booking from {booked_start_dt} to {booked_end_dt}.",
                    )
            except Exception:
                continue

    booking_id = make_booking_id(request.station_id, request.user_id, start_ts)

    # Write booking to InfluxDB, cancelled = False by default
    
    point = (
        Point("booking")
        .tag("station_id", request.station_id)
        .tag("user_id", request.user_id)
        .field("slot_start", int(start_ts))
        .field("slot_end", int(end_ts))
        .field("cancelled", False)
        .time(request.slot_start)
    )
    try:
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving booking: {str(e)}")

    return BookingResponse(
        id=booking_id,
        station_id=request.station_id,
        user_id=request.user_id,
        slot_start=request.slot_start,
        slot_end=request.slot_end,
        cancelled=False,
    )


@router.get("/user/{user_id}", response_model=List[BookingResponse])

async def get_user_bookings(
    user_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=500)
):
    # Prepare time range with proper Flux syntax
    if start:
        start_str = f'time(v: "{start.isoformat()}")'
    else:
        start_str = "-90d"

    if end:
        end_str = f'time(v: "{end.isoformat()}")'
    else:
        end_str = "now()"

    query = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r["_measurement"] == "booking")
          |> filter(fn: (r) => r["user_id"] == "{user_id}")
          |> filter(fn: (r) => r["_field"] == "slot_start" or r["_field"] == "slot_end" or r["_field"] == "cancelled")
          |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
          |> filter(fn: (r) => exists r.slot_start and exists r.slot_end and exists r.cancelled)
          |> filter(fn: (r) => r.cancelled == false)
          |> sort(columns: ["slot_start"])
          |> limit(n:{limit})
    '''
    try:
        tables = query_api.query(query, org=INFLUXDB_ORG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

    bookings = []
    for table in tables:
        points_by_time = {}
    for record in table.records:
        # Sari peste rândurile fără '_value'
        if '_value' not in record.values:
            continue
        time_key = record.get_time()
        points_by_time.setdefault(time_key, {})
        points_by_time[time_key][record.get_field()] = record.get_value()
        points_by_time[time_key]["station_id"] = record.values.get("station_id")
        points_by_time[time_key]["user_id"] = record.values.get("user_id")


        for time_key, fields in points_by_time.items():
            try:
                slot_start = datetime.utcfromtimestamp(fields["slot_start"])
                slot_end = datetime.utcfromtimestamp(fields["slot_end"])
                bookings.append(
                    BookingResponse(
                        id=make_booking_id(fields["station_id"], fields["user_id"], fields["slot_start"]),
                        station_id=fields["station_id"],
                        user_id=fields["user_id"],
                        slot_start=slot_start,
                        slot_end=slot_end,
                        cancelled=False,
                    )
                )
            except Exception:
                continue

    bookings.sort(key=lambda b: b.slot_start)
    return bookings


@router.get("/station/{station_id}", response_model=List[BookingResponse])
async def get_station_bookings(
    station_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=500)
):
    # Prepare time range with proper Flux syntax
    if start:
        start_str = f'time(v: "{start.isoformat()}")'
    else:
        start_str = "-90d"

    if end:
        end_str = f'time(v: "{end.isoformat()}")'
    else:
        end_str = "now()"

    query = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r["_measurement"] == "booking")
          |> filter(fn: (r) => r["station_id"] == "{station_id}")
          |> filter(fn: (r) => r["_field"] == "slot_start" or r["_field"] == "slot_end" or r["_field"] == "cancelled")
          |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
          |> filter(fn: (r) => exists r.slot_start and exists r.slot_end and exists r.cancelled)
          |> filter(fn: (r) => r.cancelled == false)
          |> sort(columns: ["slot_start"])
          |> limit(n:{limit})
    '''
    try:
        tables = query_api.query(query, org=INFLUXDB_ORG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

    bookings = []
    for table in tables:
        points_by_time = {}
        for record in table.records:
            time_key = record.get_time()
            points_by_time.setdefault(time_key, {})
            points_by_time[time_key][record.get_field()] = record.get_value()
            points_by_time[time_key]["station_id"] = record.values.get("station_id")
            points_by_time[time_key]["user_id"] = record.values.get("user_id")

        for time_key, fields in points_by_time.items():
            try:
                slot_start = datetime.utcfromtimestamp(fields["slot_start"])
                slot_end = datetime.utcfromtimestamp(fields["slot_end"])
                bookings.append(
                    BookingResponse(
                        id=make_booking_id(fields["station_id"], fields["user_id"], fields["slot_start"]),
                        station_id=fields["station_id"],
                        user_id=fields["user_id"],
                        slot_start=slot_start,
                        slot_end=slot_end,
                        cancelled=False,
                    )
                )
            except Exception:
                continue

    bookings.sort(key=lambda b: b.slot_start)
    return bookings


@router.post("/cancel")
async def cancel_booking(request: CancelRequest):
    station_id = request.station_id
    user_id = request.user_id
    slot_start = request.slot_start

    cancel_point = (
        Point("booking")
        .tag("station_id", station_id)
        .tag("user_id", user_id)
        .field("slot_start", int(slot_start.timestamp()))
        .field("slot_end", 0)
        .field("cancelled", True)
        .time(slot_start)
    )

    try:
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=cancel_point)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling booking: {str(e)}")

    return {"status": "success", "message": "Booking cancelled (soft delete)."}
    """
    Soft-cancel a booking by writing a duplicate record with cancelled=True.
    Actual point deletions in InfluxDB are difficult.
    """
    cancel_point = (
        Point("booking")
        .tag("station_id", station_id)
        .tag("user_id", user_id)
        .field("slot_start", int(slot_start.timestamp()))
        .field("slot_end", 0)
        .field("cancelled", True)
        .time(slot_start)
    )
    try:
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=cancel_point)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling booking: {str(e)}")

    return {"status": "success", "message": "Booking cancelled (soft delete)."}
