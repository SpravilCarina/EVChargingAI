from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --------- InfluxDB -----------
    INFLUX_URL: str = "http://localhost:8086"
    INFLUX_TOKEN: str = "WDo6QMstKcq2TvrA355ST8B7wowGyDEafy2R_C1rZmd_WqB5nRrYYCsFLmDt-NnT0URHhGGx1mUcYJNvJDsgHQ=="
    INFLUX_ORG: str = "ev-charging-ai"
    INFLUX_BUCKET: str = "incarcare_ev"

    # --------- General API -----------
    API_TITLE: str = "EV Charging AI API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Backend REST API pentru managementul stațiilor de încărcare EV"

    # --------- Alte configurări opționale (tip best-practice) -----------
    # DEBUG: bool = False
    # LOG_LEVEL: str = "info"
    # SECRET_KEY: str = "schimba-aceasta-valoare" # Pentru JWT/autentificare dacă vei folosi
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
    # DATABASE_URL: str = "sqlite:///./dev.sqlite"
    # ALLOWED_ORIGINS: str = "*"
    # SMTP_SERVER: str = "smtp.example.com"
    # SMTP_PORT: int = 587

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instanță globală pentru folosire în tot proiectul (import: from app.config import settings)
settings = Settings()
