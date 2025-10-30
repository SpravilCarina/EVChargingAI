from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importăm routere din api/
from app.api.stations import router as stations_router
from app.api.bookings import router as bookings_router
from app.api.maintenance import router as maintenance_router
from app.api.history import router as history_router
from app.api.explanation import router as explanation_router
from app.api.dynamic_price import router as dynamic_price_router
from app.api.login import router as login_router


app = FastAPI(
    title="EV Charging AI API",
    description="Backend REST API pentru managementul stațiilor de încărcare EV",
    version="1.0.0"
)

# CORS – adaptează allow_origins pe producție!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sau ["http://localhost:3000"] pentru frontend local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Înregistrăm routerele, fiecare cu prefix și tag propriu
app.include_router(stations_router, prefix="/stations", tags=["Stations"])
app.include_router(bookings_router, prefix="/bookings", tags=["Bookings"])
app.include_router(maintenance_router, prefix="/maintenance", tags=["Maintenance"])
app.include_router(history_router, prefix="/history", tags=["History"])
app.include_router(explanation_router, prefix="/explanation", tags=["Explanation"])
app.include_router(dynamic_price_router, prefix="/dynamic_price", tags=["Dynamic Price"])
app.include_router(login_router, prefix="/users", tags=["Users"])


# Endpoint root de health-check
@app.get("/")
async def root():
    return {"message": "EV Charging AI API este activ!"}
import logging
logger = logging.getLogger("uvicorn")

@app.on_event("startup")
async def startup_event():
    logger.info("EV Charging AI API has started!")
