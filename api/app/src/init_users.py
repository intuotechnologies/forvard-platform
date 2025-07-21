#!/usr/bin/env python3
import psycopg2
import uuid
from passlib.context import CryptContext
import argparse
import sys
import os
import pandas as pd

# Setup password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """Genera un hash della password usando bcrypt"""
    return pwd_context.hash(password)

def ensure_roles_exist(cur):
    """
    Assicura che i ruoli 'base', 'senator', e 'admin' esistano nella tabella roles.
    Questo risolve l'errore "Nessun ruolo trovato".
    """
    print("Assicurando l'esistenza dei ruoli...")
    roles_to_ensure = ['base', 'senator', 'admin']
    for role_name in roles_to_ensure:
        cur.execute(
            "INSERT INTO roles (role_name) VALUES (%s) ON CONFLICT (role_name) DO NOTHING;",
            (role_name,)
        )
    print("Ruoli 'base', 'senator', 'admin' verificati/creati.")

def create_users(cur):
    """
    Crea gli utenti di default se non esistono già.
    """
    print("Creazione degli utenti di default...")
    users_to_create = [
        {"email": "base@example.com", "password": "basepass", "role": "base"},
        {"email": "senator@example.com", "password": "senatorpass", "role": "senator"},
        {"email": "admin@example.com", "password": "adminpass", "role": "admin"}
    ]

    # Ottieni la mappa dei ruoli
    roles_map = {}
    cur.execute("SELECT role_id, role_name FROM roles;")
    for role_id, role_name in cur.fetchall():
        roles_map[role_name] = role_id

    if not roles_map:
        print("ERRORE CRITICO: La mappa dei ruoli è vuota anche dopo il tentativo di creazione.")
        sys.exit(1)

    for user_data in users_to_create:
        cur.execute("SELECT user_id FROM users WHERE email = %s;", (user_data['email'],))
        if cur.fetchone():
            print(f"L'utente {user_data['email']} esiste già, saltato.")
            continue

        role_id = roles_map.get(user_data['role'])
        if not role_id:
            print(f"ERRORE: Ruolo '{user_data['role']}' non trovato. Utente non creato.")
            continue

        password_hash = get_password_hash(user_data['password'])
        
        cur.execute(
            "INSERT INTO users (email, password_hash, role_id) VALUES (%s, %s, %s)",
            (user_data['email'], password_hash, role_id)
        )
        print(f"Utente {user_data['email']} creato con successo.")
    
    print("\nCredenziali degli utenti creati:")
    for user in users_to_create:
        print(f"- Email: {user['email']}, Password: {user['password']}, Ruolo: {user['role']}")


def load_assets_from_excel(cur):
    """
    Carica i dati degli asset dal file Excel nella tabella 'assets'.
    """
    print("\nCaricamento degli asset dal file Excel...")
    excel_file = 'app/sample_data/lista_asset_rv.xlsx'
    
    if not os.path.exists(excel_file):
        print(f"ERRORE: File Excel '{excel_file}' non trovato.")
        return
    
    sheet_mapping = {
        'STOCKS': 'stocks',
        'ETF': 'etf', 
        'FOREX': 'forex',
        'FUTURES': 'futures'
    }
    
    assets_to_insert = []
    
    for sheet_name, category in sheet_mapping.items():
        try:
            print(f"Lettura del foglio '{sheet_name}'...")
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            print(f"Letti {len(df)} asset dal foglio {sheet_name}")
            
            for _, row in df.iterrows():
                # Handle different column structures for different sheets
                symbol = row.get('Symbol', '')
                if not symbol:
                    continue
                    
                description = row.get('Description', '')
                industry = row.get('Industry', '')
                sector = row.get('Sector', '')
                
                # For ETF sheet, handle the Category column
                if category == 'etf' and 'Category' in row:
                    sector = row.get('Category', sector)
                
                assets_to_insert.append({
                    'symbol': symbol,
                    'description': description,
                    'industry': industry,
                    'sector': sector,
                    'asset_category': category
                })
                
        except Exception as e:
            print(f"Errore durante la lettura del foglio {sheet_name}: {e}")
            continue

    if not assets_to_insert:
        print("Nessun asset da inserire. Controlla il file Excel.")
        return

    # Inserimento bulk
    insert_query = """
        INSERT INTO assets (symbol, description, industry, sector, asset_category)
        VALUES (%(symbol)s, %(description)s, %(industry)s, %(sector)s, %(asset_category)s)
        ON CONFLICT (symbol) DO NOTHING;
    """
    cur.executemany(insert_query, assets_to_insert)
    print(f"Inseriti/aggiornati {len(assets_to_insert)} asset nel database.")


def assign_permissions(cur):
    """
    Assegna i permessi sugli asset agli utenti in base al loro ruolo.
    Regole aggiornate:
    - STOCKS: prime 40 per base, tutte per senator/admin
    - ETF: nessuna per base, tutte per senator/admin
    - FOREX: prime 5 per base, tutte per senator/admin
    - FUTURES: prime 4 per base, tutte per senator/admin
    """
    print("\nAssegnazione dei permessi...")

    # 1. Ottieni tutti gli utenti con i loro ruoli
    cur.execute("""
        SELECT u.user_id, r.role_name
        FROM users u
        JOIN roles r ON u.role_id = r.role_id;
    """)
    users = cur.fetchall()
    if not users:
        print("Nessun utente trovato per assegnare i permessi.")
        return

    # 2. Ottieni tutti gli asset per categoria
    cur.execute("""
        SELECT asset_id, symbol, asset_category 
        FROM assets 
        ORDER BY asset_category, asset_id;
    """)
    all_assets = cur.fetchall()
    if not all_assets:
        print("Nessun asset trovato per assegnare i permessi.")
        return

    # Organizza gli asset per categoria
    assets_by_category = {
        'stocks': [],
        'etf': [],
        'forex': [],
        'futures': []
    }
    
    for asset_id, symbol, category in all_assets:
        if category in assets_by_category:
            assets_by_category[category].append((asset_id, symbol))
    
    permissions_to_insert = []

    for user_id, role_name in users:
        asset_ids_to_assign = set()

        if role_name == 'base':
            # STOCKS: prime 40
            stocks_to_assign = assets_by_category['stocks'][:40]
            for asset_id, symbol in stocks_to_assign:
                asset_ids_to_assign.add(asset_id)
            
            # ETF: nessuno (non aggiungiamo nulla)
            
            # FOREX: primi 5
            forex_to_assign = assets_by_category['forex'][:5]
            for asset_id, symbol in forex_to_assign:
                asset_ids_to_assign.add(asset_id)
            
            # FUTURES: primi 4
            futures_to_assign = assets_by_category['futures'][:4]
            for asset_id, symbol in futures_to_assign:
                asset_ids_to_assign.add(asset_id)
                
            print(f"Utente base {user_id}: {len(asset_ids_to_assign)} asset assegnati "
                  f"(STOCKS: {len(stocks_to_assign)}, ETF: 0, "
                  f"FOREX: {len(forex_to_assign)}, FUTURES: {len(futures_to_assign)})")

        elif role_name in ['senator', 'admin']:
            # Assegna tutti gli asset di tutte le categorie
            for category_assets in assets_by_category.values():
                for asset_id, symbol in category_assets:
                    asset_ids_to_assign.add(asset_id)
            
            print(f"Utente {role_name} {user_id}: TUTTI i {len(asset_ids_to_assign)} asset assegnati")

        # Prepara i dati per l'inserimento
        for asset_id in asset_ids_to_assign:
            permissions_to_insert.append({'user_id': user_id, 'asset_id': asset_id})

    # 4. Inserisci tutti i permessi in un'unica operazione
    if permissions_to_insert:
        insert_query = """
            INSERT INTO asset_permissions (user_id, asset_id)
            VALUES (%(user_id)s, %(asset_id)s)
            ON CONFLICT (user_id, asset_id) DO NOTHING;
        """
        cur.executemany(insert_query, permissions_to_insert)
        print(f"Inseriti/aggiornati {len(permissions_to_insert)} permessi.")
    
    # Stampa riepilogo finale
    print("\nRiepilogo permessi per categoria:")
    for category, assets in assets_by_category.items():
        print(f"- {category.upper()}: {len(assets)} asset totali")


def initialize_database(db_host, db_port, db_name, db_user, db_password):
    """
    Funzione principale che orchestra l'inizializzazione del database.
    """
    conn = None
    try:
        print(f"Connessione al database PostgreSQL {db_host}:{db_port}...")
        conn = psycopg2.connect(
            host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password
        )
        cur = conn.cursor()
        print("Connessione stabilita con successo.")

        # Esegui i passaggi in ordine
        ensure_roles_exist(cur)
        create_users(cur)
        load_assets_from_excel(cur)  # Cambiato da load_assets_from_csv
        assign_permissions(cur)

        # Commit finale
        conn.commit()
        print("\nInizializzazione del database completata con successo!")

    except psycopg2.Error as e:
        print(f"\nERRORE durante l'operazione sul database: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Connessione al database chiusa.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inizializza il database con ruoli, utenti, asset e permessi.')
    parser.add_argument('--host', default='volare.unime.it', help='Host del database')
    parser.add_argument('--port', default='5432', help='Porta del database')
    parser.add_argument('--dbname', default='forvarddb_dev', help='Nome del database')
    parser.add_argument('--user', default='forvarduser', help='Utente del database')
    parser.add_argument('--password', default='WsUpwXjEA7HHidmL8epF', help='Password del database')
    
    args = parser.parse_args()
    
    initialize_database(
        db_host=args.host,
        db_port=args.port,
        db_name=args.dbname,
        db_user=args.user,
        db_password=args.password
    )