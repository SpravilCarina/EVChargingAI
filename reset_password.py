import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta
from typing import Optional
import jwt  # PyJWT
import bcrypt
import os

router = APIRouter()

# === Configurații/variabile de mediu (schimbă după nevoie) ===
SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_key")  # Pune un secret puternic în producție
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Token expiră în 60 minute
FRONTEND_RESET_PASSWORD_URL = os.getenv("FRONTEND_RESET_PASSWORD_URL", "http://localhost:3000/reset-password")

# === Logger config ===
logger = logging.getLogger("reset_password")
logging.basicConfig(level=logging.INFO)

# === Dummy DB (înlocuiește cu acces la baza ta reală) ===
users_db = {
    "user1@example.com": {
        "username": "user1",
        "email": "user1@example.com",
        "hashed_password": bcrypt.hashpw("parola123".encode(), bcrypt.gensalt()),
    }
}

# === Rate limiting simplificat in-memory (pentru producție folosește Redis sau alt cache distribuit) ===
rate_limit_store = {}
RATE_LIMIT_MAX_REQUESTS = 5
RATE_LIMIT_WINDOW_SECONDS = 60 * 10  # 10 minute

# === Pydantic Models ===

class PasswordResetRequest(BaseModel):
    email: EmailStr

    @validator("email")
    def email_lower(cls, v):
        return v.lower()

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

    @validator("new_password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Parola nouă trebuie să aibă cel puțin 8 caractere.")
        if not any(c.isupper() for c in v):
            raise ValueError("Parola trebuie să conțină cel puțin o literă mare.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Parola trebuie să conțină cel puțin o cifră.")
        return v

# === Helper functions ===

def create_reset_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": email, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_reset_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except jwt.PyJWTError:
        return None

def send_reset_email(email: str, token: str):
    reset_link = f"{FRONTEND_RESET_PASSWORD_URL}?token={token}"
    # TODO: Înlocuiește acest logger cu funcția reală de trimitere email (SMTP, SendGrid, Mailgun etc)
    logger.info(f"[EMAIL] Trimitem link resetare către {email}: {reset_link}")

def is_rate_limited(email: str) -> bool:
    now = datetime.utcnow()
    timestamps = rate_limit_store.get(email, [])
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)
    recent_requests = [ts for ts in timestamps if ts > window_start]
    rate_limit_store[email] = recent_requests  # actualizează lista cererilor valide
    if len(recent_requests) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    rate_limit_store[email].append(now)
    return False

# === Endpoints ===

@router.post("/users/reset_password", tags=["Users"], summary="Cere resetare parolă")
async def reset_password_request(request: PasswordResetRequest, background_tasks: BackgroundTasks):
    email = request.email
    if is_rate_limited(email):
        logger.warning(f"Rate limit atins pentru resetare parola: {email}")
        # Pentru securitate: răspuns generic indiferent dacă email-ul există sau nu
        return {"message": "Dacă există un cont asociat acestui email, vei primi instrucțiuni de resetare a parolei."}

    user = users_db.get(email)
    if user:
        token = create_reset_token(email)
        # Trimite email asincron
        background_tasks.add_task(send_reset_email, email, token)
        logger.info(f"Resetare parolă inițiată pentru utilizatorul: {email}")
    else:
        logger.info(f"Cerere resetare parolă de la email necunoscut: {email}")

    return {"message": "Dacă există un cont asociat acestui email, vei primi instrucțiuni de resetare a parolei."}

@router.post("/users/reset_password_confirm", tags=["Users"], summary="Confirmă resetarea parolei")
async def reset_password_confirm(data: PasswordResetConfirm):
    email = verify_reset_token(data.token)
    if not email:
        logger.warning("Token invalid sau expirat la confirmarea resetării parolei")
        raise HTTPException(status_code=400, detail="Token invalid sau expirat.")

    user = users_db.get(email)
    if not user:
        logger.warning(f"Token valid, dar utilizator inexistent: {email}")
        raise HTTPException(status_code=400, detail="Utilizator inexistent.")

    # Validarea parolei este deja făcută de Pydantic validator
    hashed = bcrypt.hashpw(data.new_password.encode(), bcrypt.gensalt())
    user["hashed_password"] = hashed
    logger.info(f"Parolă resetată cu succes pentru utilizatorul: {email}")
    # TODO: Salvează modificarea în baza ta reală de date aici, cu commit

    # Opțional - trimite email de confirmare resetare parolă
    # background_tasks.add_task(send_confirmation_email, email)

    return {"message": "Parola a fost resetată cu succes."}
