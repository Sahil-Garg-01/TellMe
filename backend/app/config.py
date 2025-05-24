from pydantic import BaseModel

class Settings(BaseModel):
    authjwt_secret_key: str = "super-secret"  # Replace with a secure key

def get_settings():
    return Settings()