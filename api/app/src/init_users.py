#!/usr/bin/env python3
# init_users.py - Script per inizializzare utenti e permessi nel database
import psycopg2
import uuid
from passlib.context import CryptContext
import argparse
import sys

# Setup password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """Genera un hash della password usando bcrypt"""
    return pwd_context.hash(password)

def create_users_and_permissions(
    db_host="79.72.44.95",
    db_port="5432",
    db_name="appdb",
    db_user="appuser",
    db_password="appsecretpassword"
):
    """
    Crea utenti e permessi base nel database
    """
    conn = None
    cur = None

    # Definizioni utenti
    users = [
        {"email": "base@example.com", "password": "basepass", "role": "base"},
        {"email": "senator@example.com", "password": "senatorpass", "role": "senator"},
        {"email": "admin@example.com", "password": "adminpass", "role": "admin"}
    ]

    try:
        # Connessione al database
        print(f"Connessione al database PostgreSQL {db_host}:{db_port}...")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        cur = conn.cursor()
        print("Connessione stabilita con successo.")

        # Verifica che le tabelle necessarie esistano
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'users'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("ERRORE: La tabella 'users' non esiste nel database.")
            print("Assicurati che lo schema del database sia stato correttamente inizializzato.")
            sys.exit(1)

        # Ottieni gli ID dei ruoli esistenti
        roles = {}
        cur.execute("SELECT role_id, role_name FROM roles;")
        for role_id, role_name in cur.fetchall():
            roles[role_name] = role_id

        if not roles:
            print("ERRORE: Nessun ruolo trovato nella tabella 'roles'.")
            print("Assicurati che lo schema del database sia stato correttamente inizializzato.")
            sys.exit(1)

        # Crea gli utenti
        for user in users:
            # Controlla se l'utente esiste già
            cur.execute("SELECT user_id FROM users WHERE email = %s;", (user['email'],))
            existing_user = cur.fetchone()
            
            if existing_user:
                print(f"L'utente {user['email']} esiste già, verrà saltato.")
                continue
            
            user_id = str(uuid.uuid4())
            role_id = roles.get(user['role'])
            
            if not role_id:
                print(f"ERRORE: Ruolo '{user['role']}' non trovato nel database. Utente {user['email']} non creato.")
                continue
            
            password_hash = get_password_hash(user['password'])
            
            # Inserisci l'utente
            cur.execute("""
                INSERT INTO users (user_id, email, password_hash, role_id)
                VALUES (%s, %s, %s, %s)
            """, (user_id, user['email'], password_hash, role_id))
            
            print(f"Utente {user['email']} creato con successo (ruolo: {user['role']}).")
        
        # Commit delle modifiche
        conn.commit()
        print("\nCreazione utenti completata con successo.")
        print("\nCredenziali degli utenti creati:")
        for user in users:
            print(f"- Email: {user['email']}, Password: {user['password']}, Ruolo: {user['role']}")
        
    except psycopg2.Error as e:
        print(f"Errore durante l'operazione sul database: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Connessione al database chiusa.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inizializza utenti e permessi nel database')
    parser.add_argument('--host', default='79.72.44.95', help='Host del database')
    parser.add_argument('--port', default='5432', help='Porta del database')
    parser.add_argument('--dbname', default='appdb', help='Nome del database')
    parser.add_argument('--user', default='appuser', help='Utente del database')
    parser.add_argument('--password', default='appsecretpassword', help='Password del database')
    
    args = parser.parse_args()
    
    create_users_and_permissions(
        db_host=args.host,
        db_port=args.port,
        db_name=args.dbname,
        db_user=args.user,
        db_password=args.password
    ) 