from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
from .auth import router as auth_router
from .memory import router as memory_router
# from fastapi_jwt_auth import AuthJWT  # Removed, no longer needed
from dotenv import load_dotenv
from .database import get_db
import os

load_dotenv()

app = FastAPI()
app.include_router(auth_router, prefix="/auth")
app.include_router(memory_router, prefix="/memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Removed AuthJWT.load_config and related settings, as JWT is now handled with python-jose

@app.get("/")
def read_root():
    return {"message": "AI Text Memorizer API is running"}
