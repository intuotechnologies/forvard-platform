import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import Depends, HTTPException, status
from loguru import logger
import time
import pandas as pd

# Get database URL from environment variables with fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL_API", 
    "postgresql://forvarduser:WsUpwXjEA7HHidmL8epF@volare.unime.it:5432/forvarddb_dev"
)

# Create SQLAlchemy engine and session
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info(f"Database connection initialized to {DATABASE_URL.split('@')[1]}")
except SQLAlchemyError as e:
    logger.error(f"Failed to initialize database connection: {e}")
    raise


def get_db():
    """
    Dependency to get a database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def check_db_connection(max_retries: int = 5, retry_interval: int = 5) -> bool:
    """
    Check database connection with retry mechanism
    """
    for attempt in range(max_retries):
        try:
            with SessionLocal() as db:
                db.execute(text("SELECT 1"))
                logger.info("Database connection successful")
                return True
        except SQLAlchemyError as e:
            logger.warning(f"Database connection attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
    
    logger.error("All database connection attempts failed")
    return False


def query_to_dataframe(db: Session, query: str, params: dict = None) -> pd.DataFrame:
    """
    Execute SQL query and return results as pandas DataFrame
    """
    try:
        result = db.execute(text(query), params or {})
        columns = result.keys()
        return pd.DataFrame(result.fetchall(), columns=columns)
    except SQLAlchemyError as e:
        logger.error(f"SQL query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database query error"
        ) 