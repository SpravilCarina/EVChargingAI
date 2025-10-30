from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

# Configurări generale — modifică SECRET_KEY cu o valoare sigură în producție!
SECRET_KEY = "un-secret-key-lung-si-random-foarte-securizat-schimba-l"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Routerul API
router = APIRouter()

# Simulare "bază de date" în memorie (user: hashed_password)
fake_users_db: Dict[str, str] = {}

# Configurare hashing parole
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


# Scheme OAuth2 pentru password flow (token URL)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")

# Modele Pydantic
class UserAuth(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Funcții utilitare pentru hashing și token
def hash_password(password: str) -> str:
    max_length = 72
    truncated_password = password[:max_length]  # taie parola dacă e prea lungă
    return pwd_context.hash(truncated_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Dependență pentru obținerea userului curent validat prin token
async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Nu s-a putut valida autentificarea tokenului",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in fake_users_db:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception

# Endpointuri API

@router.post("/register", summary="Înregistrare user nou")
async def register(user: UserAuth):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="User deja există")
    hashed_password = hash_password(user.password)
    fake_users_db[user.username] = hashed_password
    return {"status": "cont creat cu succes"}

@router.post("/login", response_model=Token, summary="Login user existent")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password
    hashed_password = fake_users_db.get(username)
    if not hashed_password or not verify_password(password, hashed_password):
        raise HTTPException(status_code=401, detail="Autentificare eșuată: user/parolă invalidă", headers={"WWW-Authenticate": "Bearer"})
    
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", summary="Utilizator curent", response_model=str)
async def read_current_user(current_user: str = Depends(get_current_user)):
    return current_user
