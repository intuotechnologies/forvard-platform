import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text

from app.main import app
from app.core.database import get_db
from app.core.security import get_password_hash

# Use in-memory SQLite for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

@pytest.fixture
def db_session():
    """
    Create a clean database session for running tests.
    Returns a SQLAlchemy session with tables created.
    """
    # Setup the database
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    # Create tables
    with open('tests/fixtures/create_test_db.sql', 'r') as f:
        session.execute(text(f.read()))
        session.commit()

    # Insert test data
    with open('tests/fixtures/insert_test_data.sql', 'r') as f:
        session.execute(text(f.read()))
        session.commit()

    yield session

    # Rollback the transaction and close the connection
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session):
    """
    Create a FastAPI TestClient with the test database session.
    """
    # Override the get_db dependency to use our test database
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    
    # Create test client
    with TestClient(app) as client:
        yield client
        
    # Clean up - reset dependency overrides
    app.dependency_overrides = {}

@pytest.fixture
def test_admin_token(client):
    """Get a valid admin token for testing protected endpoints"""
    response = client.post(
        "/auth/token",
        data={"username": "admin@example.com", "password": "adminpass"}
    )
    return response.json()["access_token"]

@pytest.fixture
def test_senator_token(client):
    """Get a valid senator user token for testing protected endpoints with limited access"""
    response = client.post(
        "/auth/token",
        data={"username": "senator@example.com", "password": "senatorpass"}
    )
    return response.json()["access_token"]

@pytest.fixture
def test_base_token(client):
    """Get a valid base user token for testing protected endpoints with basic access"""
    response = client.post(
        "/auth/token",
        data={"username": "user@example.com", "password": "userpass"}
    )
    return response.json()["access_token"] 