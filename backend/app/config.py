from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    GOOGLE_CLIENT_ID: str
    JWT_SECRET: str
    JWT_EXPIRE_MINUTES: int = 720
    FRONTEND_ORIGIN: str = "*"
    FACE_MATCH_TOLERANCE: float = 0.5

    class Config:
        env_file = ".env"


settings = Settings()
