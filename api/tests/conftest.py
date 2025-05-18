import pytest
import os
import re
import sqlite3
import uuid
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text

from app.main import app
from app.core.database import get_db
from app.core.security import get_password_hash

# Usa sempre SQLite per i test, sia in locale che nella CI
# In questo modo garantiamo consistenza nei test tra i diversi ambienti
TEST_DB_PATH = os.path.abspath("test.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{TEST_DB_PATH}"

# Configurazione specifica per SQLite
# Registra UUID converter
sqlite3.register_converter("GUID", lambda b: uuid.UUID(bytes_le=b))
sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes_le)

# Configura l'engine SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Attiva foreign keys per SQLite
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

@pytest.fixture(scope="session")
def db_session():
    """
    Create a clean database session for testing with SQLite
    """
    # Rimuovi vecchio database se esiste
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    
    # Collegati direttamente a SQLite per eseguire script SQL
    conn = sqlite3.connect(TEST_DB_PATH)
    
    # Carica script di creazione schema
    schema_path = os.path.join(os.path.dirname(__file__), "fixtures", "create_test_db.sql")
    with open(schema_path, "r") as f:
        schema_script = f.read()
    
    # Esegui ogni statement nello script schema
    for statement in schema_script.split(';'):
        if statement.strip():
            conn.execute(statement)
    
    # Inserisci dati di test con password hash corrette
    # Dati ruolo
    conn.execute("""
    INSERT INTO roles (role_id, role_name, description) 
    VALUES 
        (1, 'admin', 'Administrator with full access'),
        (2, 'senator', 'Senator with extended access'),
        (3, 'base', 'Base user with limited access')
    """)
    
    # Dati utente con hash di password corretti per i test
    admin_pass_hash = get_password_hash("adminpass")
    senator_pass_hash = get_password_hash("senatorpass")
    user_pass_hash = get_password_hash("userpass")
    
    conn.execute(
        "INSERT INTO users (user_id, email, password_hash, role_id) VALUES (?, ?, ?, ?)",
        (str(uuid.uuid4()), "admin@example.com", admin_pass_hash, 1)
    )
    conn.execute(
        "INSERT INTO users (user_id, email, password_hash, role_id) VALUES (?, ?, ?, ?)",
        (str(uuid.uuid4()), "senator@example.com", senator_pass_hash, 2)
    )
    conn.execute(
        "INSERT INTO users (user_id, email, password_hash, role_id) VALUES (?, ?, ?, ?)",
        (str(uuid.uuid4()), "user@example.com", user_pass_hash, 3)
    )
    
    # Inserisci limiti di accesso 
    conn.execute("""
    INSERT INTO asset_access_limits (limit_id, role_id, asset_category, max_items) 
    VALUES 
        (1, 1, 'equity', 100),
        (2, 1, 'fx', 100),
        (3, 1, 'crypto', 100),
        (4, 2, 'equity', 50),
        (5, 2, 'fx', 50),
        (6, 2, 'crypto', 50),
        (7, 3, 'equity', 10),
        (8, 3, 'fx', 10),
        (9, 3, 'crypto', 10)
    """)
    
    # Inserisci dati di volatilità campione
    with open(os.path.join(os.path.dirname(__file__), "fixtures", "sample_data.sql"), "r") as f:
        sample_data = f.read()
        conn.executescript(sample_data)
    
    conn.commit()
    conn.close()
    
    # Crea una sessione SQLAlchemy di test
    try:
        # Override della dipendenza get_db
        def override_get_db():
            db = TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()
                
        # Sostituisci get_db originale con la versione di test
        app.dependency_overrides[get_db] = override_get_db
        
        # Restituisci una sessione per il test
        db = TestingSessionLocal()
        yield db
        db.close()
    finally:
        # Pulisci dopo i test
        app.dependency_overrides.clear()


@pytest.fixture
def client(db_session):
    """Test client for FastAPI app"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_admin_token(client):
    """Get a valid admin token for testing protected endpoints"""
    response = client.post(
        "/auth/token",
        data={"username": "admin@example.com", "password": "adminpass"}
    )
    token_data = response.json()
    assert "access_token" in token_data, f"Failed to get token: {response.text}"
    return token_data["access_token"]


@pytest.fixture
def test_senator_token(client):
    """Get a valid senator user token for testing protected endpoints with limited access"""
    response = client.post(
        "/auth/token",
        data={"username": "senator@example.com", "password": "senatorpass"}
    )
    token_data = response.json()
    assert "access_token" in token_data, f"Failed to get token: {response.text}"
    return token_data["access_token"]


@pytest.fixture
def test_base_token(client):
    """Get a valid base user token for testing protected endpoints with basic access"""
    response = client.post(
        "/auth/token",
        data={"username": "user@example.com", "password": "userpass"}
    )
    token_data = response.json()
    assert "access_token" in token_data, f"Failed to get token: {response.text}"
    return token_data["access_token"]
