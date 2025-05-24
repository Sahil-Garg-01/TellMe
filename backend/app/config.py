from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

class Settings(BaseModel):
    authjwt_secret_key: str = os.getenv("JWT_SECRET_KEY")

def get_settings():
    return Settings()
