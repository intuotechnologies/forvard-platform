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


def load_assets_from_data(cur):
    """
    Carica i dati degli asset specifici nella tabella 'assets'.
    """
    print("\nCaricamento degli asset specifici...")
    
    # Definizione degli asset specifici
    assets_data = {
        'stocks': [
            {'symbol': 'AAPL', 'description': 'APPLE INC. COMMON STOCK', 'industry': 'Computer Manufacturing', 'sector': 'Technology'},
            {'symbol': 'ABBV', 'description': 'ABBVIE INC. COMMON STOCK', 'industry': 'Biotechnology: Pharmaceutical Preparations', 'sector': 'Health Care'},
            {'symbol': 'ABT', 'description': 'ABBOTT LABORATORIES COMMON STOCK', 'industry': 'Biotechnology: Pharmaceutical Preparations', 'sector': 'Health Care'},
            {'symbol': 'ACN', 'description': 'ACCENTURE PLC CLASS A ORDINARY SHARES (IRELAND)', 'industry': 'Business Services', 'sector': 'Consumer Discretionary'},
            {'symbol': 'ADBE', 'description': 'ADOBE INC. COMMON STOCK', 'industry': 'Computer Software: Prepackaged Software', 'sector': 'Technology'},
            {'symbol': 'AIG', 'description': 'AMERICAN INTERNATIONAL GROUP INC. NEW COMMON STOCK', 'industry': 'Life Insurance', 'sector': 'Finance'},
            {'symbol': 'AMD', 'description': 'ADVANCED MICRO DEVICES INC. COMMON STOCK', 'industry': 'Semiconductors', 'sector': 'Technology'},
            {'symbol': 'AMGN', 'description': 'AMGEN INC. COMMON STOCK', 'industry': 'Biotechnology: Biological Products (No Diagnostic Substances)', 'sector': 'Health Care'},
            {'symbol': 'AMT', 'description': 'AMERICAN TOWER CORPORATION (REIT) COMMON STOCK', 'industry': 'Real Estate Investment Trusts', 'sector': 'Real Estate'},
            {'symbol': 'AMZN', 'description': 'AMAZON.COM INC. COMMON STOCK', 'industry': 'Catalog/Specialty Distribution', 'sector': 'Consumer Discretionary'},
            {'symbol': 'AVGO', 'description': 'BROADCOM INC. COMMON STOCK', 'industry': 'Semiconductors', 'sector': 'Technology'},
            {'symbol': 'AXP', 'description': 'AMERICAN EXPRESS COMPANY COMMON STOCK', 'industry': 'Finance: Consumer Services', 'sector': 'Finance'},
            {'symbol': 'BA', 'description': 'BOEING COMPANY (THE) COMMON STOCK', 'industry': 'Aerospace', 'sector': 'Industrials'},
            {'symbol': 'BAC', 'description': 'BANK OF AMERICA CORPORATION COMMON STOCK', 'industry': 'Major Banks', 'sector': 'Finance'},
            {'symbol': 'BK', 'description': 'THE BANK OF NEW YORK MELLON CORPORATION COMMON STOCK', 'industry': 'Major Banks', 'sector': 'Finance'},
            {'symbol': 'BKNG', 'description': 'BOOKING HOLDINGS INC. COMMON STOCK', 'industry': 'Transportation Services', 'sector': 'Consumer Discretionary'},
            {'symbol': 'BLK', 'description': 'BLACKROCK INC. COMMON STOCK', 'industry': 'Investment Bankers/Brokers/Service', 'sector': 'Finance'},
            {'symbol': 'BMY', 'description': 'BRISTOL-MYERS SQUIBB COMPANY COMMON STOCK', 'industry': 'Biotechnology: Pharmaceutical Preparations', 'sector': 'Health Care'},
            {'symbol': 'BRK.B', 'description': 'BERKSHIRE HATHAWAY INC.', 'industry': '', 'sector': ''},
            {'symbol': 'C', 'description': 'CITIGROUP INC. COMMON STOCK', 'industry': 'Major Banks', 'sector': 'Finance'},
            {'symbol': 'CAT', 'description': 'CATERPILLAR INC. COMMON STOCK', 'industry': 'Construction/Ag Equipment/Trucks', 'sector': 'Industrials'},
            {'symbol': 'CHTR', 'description': 'CHARTER COMMUNICATIONS INC. CLASS A COMMON STOCK NEW', 'industry': 'Cable & Other Pay Television Services', 'sector': 'Telecommunications'},
            {'symbol': 'CL', 'description': 'COLGATE-PALMOLIVE COMPANY COMMON STOCK', 'industry': 'Package Goods/Cosmetics', 'sector': 'Consumer Discretionary'},
            {'symbol': 'CMCSA', 'description': 'COMCAST CORPORATION CLASS A COMMON STOCK', 'industry': 'Cable & Other Pay Television Services', 'sector': 'Telecommunications'},
            {'symbol': 'COF', 'description': 'CAPITAL ONE FINANCIAL CORPORATION COMMON STOCK', 'industry': 'Major Banks', 'sector': 'Finance'},
            {'symbol': 'COP', 'description': 'CONOCOPHILLIPS COMMON STOCK', 'industry': 'Integrated oil Companies', 'sector': 'Energy'},
            {'symbol': 'COST', 'description': 'COSTCO WHOLESALE CORPORATION COMMON STOCK', 'industry': 'Department/Specialty Retail Stores', 'sector': 'Consumer Discretionary'},
            {'symbol': 'COIN', 'description': 'COINBASE GLOBAL INC. CLASS A COMMON STOCK', 'industry': 'Finance: Consumer Services', 'sector': 'Finance'},
            {'symbol': 'CRM', 'description': 'SALESFORCE INC. COMMON STOCK', 'industry': 'Computer Software: Prepackaged Software', 'sector': 'Technology'},
            {'symbol': 'CSCO', 'description': 'CISCO SYSTEMS INC. COMMON STOCK (DE)', 'industry': 'Computer Communications Equipment', 'sector': 'Telecommunications'},
            {'symbol': 'CVS', 'description': 'CVS HEALTH CORPORATION COMMON STOCK', 'industry': 'Retail-Drug Stores and Proprietary Stores', 'sector': 'Consumer Staples'},
            {'symbol': 'CVX', 'description': 'CHEVRON CORPORATION COMMON STOCK', 'industry': 'Integrated oil Companies', 'sector': 'Energy'},
            {'symbol': 'DE', 'description': 'DEERE & COMPANY COMMON STOCK', 'industry': 'Industrial Machinery/Components', 'sector': 'Industrials'},
            {'symbol': 'DHR', 'description': 'DANAHER CORPORATION COMMON STOCK', 'industry': 'Industrial Machinery/Components', 'sector': 'Industrials'},
            {'symbol': 'DIS', 'description': 'WALT DISNEY COMPANY (THE) COMMON STOCK', 'industry': 'Services-Misc. Amusement & Recreation', 'sector': 'Consumer Discretionary'},
            {'symbol': 'DOW', 'description': 'DOW INC. COMMON STOCK', 'industry': 'Major Chemicals', 'sector': 'Industrials'},
            {'symbol': 'DUK', 'description': 'DUKE ENERGY CORPORATION (HOLDING COMPANY) COMMON STOCK', 'industry': 'Power Generation', 'sector': 'Utilities'},
            {'symbol': 'EMR', 'description': 'EMERSON ELECTRIC COMPANY COMMON STOCK', 'industry': 'Consumer Electronics/Appliances', 'sector': 'Technology'},
            {'symbol': 'EBAY', 'description': 'EBAY INC. COMMON STOCK', 'industry': 'Business Services', 'sector': 'Consumer Discretionary'},
            {'symbol': 'F', 'description': 'FORD MOTOR COMPANY COMMON STOCK', 'industry': 'Auto Manufacturing', 'sector': 'Consumer Discretionary'},
            {'symbol': 'FDX', 'description': 'FEDEX CORPORATION COMMON STOCK', 'industry': 'Air Freight/Delivery Services', 'sector': 'Consumer Discretionary'},
            {'symbol': 'GD', 'description': 'GENERAL DYNAMICS CORPORATION COMMON STOCK', 'industry': 'Marine Transportation', 'sector': 'Industrials'},
            {'symbol': 'GE', 'description': 'GE AEROSPACE COMMON STOCK', 'industry': 'Consumer Electronics/Appliances', 'sector': 'Technology'},
            {'symbol': 'GILD', 'description': 'GILEAD SCIENCES INC. COMMON STOCK', 'industry': 'Biotechnology: Biological Products (No Diagnostic Substances)', 'sector': 'Health Care'},
            {'symbol': 'GM', 'description': 'GENERAL MOTORS COMPANY COMMON STOCK', 'industry': 'Auto Manufacturing', 'sector': 'Consumer Discretionary'},
            {'symbol': 'GOOG', 'description': 'ALPHABET INC. CLASS C CAPITAL STOCK', 'industry': 'Computer Software: Programming Data Processing', 'sector': 'Technology'},
            {'symbol': 'GOOGL', 'description': 'ALPHABET INC. CLASS A COMMON STOCK', 'industry': 'Computer Software: Programming Data Processing', 'sector': 'Technology'},
            {'symbol': 'GS', 'description': 'GOLDMAN SACHS GROUP INC. (THE) COMMON STOCK', 'industry': 'Investment Bankers/Brokers/Service', 'sector': 'Finance'},
            {'symbol': 'HD', 'description': 'HOME DEPOT INC. (THE) COMMON STOCK', 'industry': 'RETAIL: Building Materials', 'sector': 'Consumer Discretionary'},
            {'symbol': 'HON', 'description': 'HONEYWELL INTERNATIONAL INC. COMMON STOCK', 'industry': 'Aerospace', 'sector': 'Industrials'},
            {'symbol': 'IBM', 'description': 'INTERNATIONAL BUSINESS MACHINES CORPORATION COMMON STOCK', 'industry': 'Computer Manufacturing', 'sector': 'Technology'},
            {'symbol': 'INTC', 'description': 'INTEL CORPORATION COMMON STOCK', 'industry': 'Semiconductors', 'sector': 'Technology'},
            {'symbol': 'INTU', 'description': 'INTUIT INC. COMMON STOCK', 'industry': 'Computer Software: Prepackaged Software', 'sector': 'Technology'},
            {'symbol': 'JNJ', 'description': 'JOHNSON & JOHNSON COMMON STOCK', 'industry': 'Biotechnology: Pharmaceutical Preparations', 'sector': 'Health Care'},
            {'symbol': 'JPM', 'description': 'JP MORGAN CHASE & CO. COMMON STOCK', 'industry': 'Major Banks', 'sector': 'Finance'},
            {'symbol': 'KHC', 'description': 'THE KRAFT HEINZ COMPANY COMMON STOCK', 'industry': 'Packaged Foods', 'sector': 'Consumer Staples'},
            {'symbol': 'KO', 'description': 'COCA-COLA COMPANY (THE) COMMON STOCK', 'industry': 'Beverages (Production/Distribution)', 'sector': 'Consumer Staples'},
            {'symbol': 'LIN', 'description': 'LINDE PLC ORDINARY SHARES', 'industry': 'Major Chemicals', 'sector': 'Industrials'},
            {'symbol': 'LLY', 'description': 'ELI LILLY AND COMPANY COMMON STOCK', 'industry': 'Biotechnology: Pharmaceutical Preparations', 'sector': 'Health Care'},
            {'symbol': 'LMT', 'description': 'LOCKHEED MARTIN CORPORATION COMMON STOCK', 'industry': 'Military/Government/Technical', 'sector': 'Industrials'},
            {'symbol': 'LOW', 'description': 'LOWE\'S COMPANIES INC. COMMON STOCK', 'industry': 'RETAIL: Building Materials', 'sector': 'Consumer Discretionary'},
            {'symbol': 'MA', 'description': 'MASTERCARD INCORPORATED COMMON STOCK', 'industry': 'Business Services', 'sector': 'Consumer Discretionary'},
            {'symbol': 'MCD', 'description': 'MCDONALD\'S CORPORATION COMMON STOCK', 'industry': 'Restaurants', 'sector': 'Consumer Discretionary'},
            {'symbol': 'MDLZ', 'description': 'MONDELEZ INTERNATIONAL INC. CLASS A COMMON STOCK', 'industry': 'Packaged Foods', 'sector': 'Consumer Staples'},
            {'symbol': 'MDT', 'description': 'MEDTRONIC PLC. ORDINARY SHARES', 'industry': 'Biotechnology: Electromedical & Electrotherapeutic Apparatus', 'sector': 'Health Care'},
            {'symbol': 'MET', 'description': 'METLIFE INC. COMMON STOCK', 'industry': 'Life Insurance', 'sector': 'Finance'},
            {'symbol': 'META', 'description': 'META PLATFORMS INC. CLASS A COMMON STOCK', 'industry': 'Computer Software: Programming Data Processing', 'sector': 'Technology'},
            {'symbol': 'MMM', 'description': '3M COMPANY COMMON STOCK', 'industry': 'Medical/Dental Instruments', 'sector': 'Health Care'},
            {'symbol': 'MO', 'description': 'ALTRIA GROUP INC.', 'industry': 'Medicinal Chemicals and Botanical Products', 'sector': 'Health Care'},
            {'symbol': 'MRK', 'description': 'MERCK & COMPANY INC. COMMON STOCK (NEW)', 'industry': 'Biotechnology: Pharmaceutical Preparations', 'sector': 'Health Care'},
            {'symbol': 'MS', 'description': 'MORGAN STANLEY COMMON STOCK', 'industry': 'Investment Bankers/Brokers/Service', 'sector': 'Finance'},
            {'symbol': 'MSFT', 'description': 'MICROSOFT CORPORATION COMMON STOCK', 'industry': 'Computer Software: Prepackaged Software', 'sector': 'Technology'},
            {'symbol': 'NEE', 'description': 'NEXTERA ENERGY INC. COMMON STOCK', 'industry': 'EDP Services', 'sector': 'Technology'},
            {'symbol': 'NFLX', 'description': 'NETFLIX INC. COMMON STOCK', 'industry': 'Consumer Electronics/Video Chains', 'sector': 'Consumer Discretionary'},
            {'symbol': 'NKE', 'description': 'NIKE INC. COMMON STOCK', 'industry': 'Shoe Manufacturing', 'sector': 'Consumer Discretionary'},
            {'symbol': 'NVDA', 'description': 'NVIDIA CORPORATION COMMON STOCK', 'industry': 'Semiconductors', 'sector': 'Technology'},
            {'symbol': 'ORCL', 'description': 'ORACLE CORPORATION COMMON STOCK', 'industry': 'Computer Software: Prepackaged Software', 'sector': 'Technology'},
            {'symbol': 'PEP', 'description': 'PEPSICO INC. COMMON STOCK', 'industry': 'Beverages (Production/Distribution)', 'sector': 'Consumer Staples'},
            {'symbol': 'PFE', 'description': 'PFIZER INC. COMMON STOCK', 'industry': 'Biotechnology: Pharmaceutical Preparations', 'sector': 'Health Care'},
            {'symbol': 'PG', 'description': 'PROCTER & GAMBLE COMPANY (THE) COMMON STOCK', 'industry': 'Package Goods/Cosmetics', 'sector': 'Consumer Discretionary'},
            {'symbol': 'PM', 'description': 'PHILIP MORRIS INTERNATIONAL INC COMMON STOCK', 'industry': 'Medicinal Chemicals and Botanical Products', 'sector': 'Health Care'},
            {'symbol': 'PYPL', 'description': 'PAYPAL HOLDINGS INC. COMMON STOCK', 'industry': 'Business Services', 'sector': 'Consumer Discretionary'},
            {'symbol': 'QCOM', 'description': 'QUALCOMM INCORPORATED COMMON STOCK', 'industry': 'Radio And Television Broadcasting And Communications Equipment', 'sector': 'Technology'},
            {'symbol': 'RTX', 'description': 'RTX CORPORATION COMMON STOCK', 'industry': 'Aerospace', 'sector': 'Industrials'},
            {'symbol': 'SBUX', 'description': 'STARBUCKS CORPORATION COMMON STOCK', 'industry': 'Restaurants', 'sector': 'Consumer Discretionary'},
            {'symbol': 'SCHW', 'description': 'CHARLES SCHWAB CORPORATION (THE) COMMON STOCK', 'industry': 'Investment Bankers/Brokers/Service', 'sector': 'Finance'},
            {'symbol': 'SHOP', 'description': 'SHOPIFY INC. CLASS A SUBORDINATE VOTING SHARES', 'industry': 'Computer Software: Prepackaged Software', 'sector': 'Technology'},
            {'symbol': 'SNAP', 'description': 'SNAP INC. CLASS A COMMON STOCK', 'industry': 'Computer Software: Programming Data Processing', 'sector': 'Technology'},
            {'symbol': 'SO', 'description': 'SOUTHERN COMPANY (THE) COMMON STOCK', 'industry': 'Electric Utilities: Central', 'sector': 'Utilities'},
            {'symbol': 'SPG', 'description': 'SIMON PROPERTY GROUP INC. COMMON STOCK', 'industry': 'Real Estate Investment Trusts', 'sector': 'Real Estate'},
            {'symbol': 'SPOT', 'description': 'SPOTIFY TECHNOLOGY S.A. ORDINARY SHARES', 'industry': 'Broadcasting', 'sector': 'Consumer Discretionary'},
            {'symbol': 'T', 'description': 'AT&T INC.', 'industry': 'Telecommunications Equipment', 'sector': 'Telecommunications'},
            {'symbol': 'TGT', 'description': 'TARGET CORPORATION COMMON STOCK', 'industry': 'Department/Specialty Retail Stores', 'sector': 'Consumer Discretionary'},
            {'symbol': 'TMO', 'description': 'THERMO FISHER SCIENTIFIC INC COMMON STOCK', 'industry': 'Industrial Machinery/Components', 'sector': 'Industrials'},
            {'symbol': 'TMUS', 'description': 'T-MOBILE US INC. COMMON STOCK', 'industry': 'Telecommunications Equipment', 'sector': 'Telecommunications'},
            {'symbol': 'TSLA', 'description': 'TESLA INC. COMMON STOCK', 'industry': 'Auto Manufacturing', 'sector': 'Consumer Discretionary'},
            {'symbol': 'TXN', 'description': 'TEXAS INSTRUMENTS INCORPORATED COMMON STOCK', 'industry': 'Semiconductors', 'sector': 'Technology'},
            {'symbol': 'UBER', 'description': 'UBER TECHNOLOGIES INC. COMMON STOCK', 'industry': 'Business Services', 'sector': 'Consumer Discretionary'},
            {'symbol': 'UNH', 'description': 'UNITEDHEALTH GROUP INCORPORATED COMMON STOCK (DE)', 'industry': 'Medical Specialities', 'sector': 'Health Care'},
            {'symbol': 'UNP', 'description': 'UNION PACIFIC CORPORATION COMMON STOCK', 'industry': 'Railroads', 'sector': 'Industrials'},
            {'symbol': 'UPS', 'description': 'UNITED PARCEL SERVICE INC. COMMON STOCK', 'industry': 'Trucking Freight/Courier Services', 'sector': 'Industrials'},
            {'symbol': 'USB', 'description': 'U.S. BANCORP COMMON STOCK', 'industry': 'Major Banks', 'sector': 'Finance'},
            {'symbol': 'V', 'description': 'VISA INC.', 'industry': 'Business Services', 'sector': 'Consumer Discretionary'},
            {'symbol': 'VZ', 'description': 'VERIZON COMMUNICATIONS INC.', 'industry': 'Telecommunications Equipment', 'sector': 'Public Utilities'},
            {'symbol': 'WFC', 'description': 'WELLS FARGO & COMPANY COMMON STOCK', 'industry': 'Major Banks', 'sector': 'Finance'},
            {'symbol': 'WMT', 'description': 'WALMART INC. COMMON STOCK', 'industry': 'Department/Specialty Retail Stores', 'sector': 'Consumer Discretionary'},
            {'symbol': 'XOM', 'description': 'EXXON MOBIL CORPORATION COMMON STOCK', 'industry': 'Integrated oil Companies', 'sector': 'Energy'},
            {'symbol': 'ZM', 'description': 'ZOOM COMMUNICATIONS INC. CLASS A COMMON STOCK', 'industry': 'Computer Software: Programming Data Processing', 'sector': 'Technology'}
        ],
        'etf': [
            {'symbol': 'AGG', 'description': 'ISHARES CORE US AGGREGATE BOND', 'industry': 'iShares', 'sector': 'Intermediate-Term Bond'},
            {'symbol': 'BND', 'description': 'VANGUARD TOTAL BOND MARKET ETF', 'industry': 'Vanguard', 'sector': 'Intermediate-Term Bond'},
            {'symbol': 'GLD', 'description': 'SPDR GOLD TRUST', 'industry': 'State Street Global Advisors', 'sector': 'Commodities Precious Metals'},
            {'symbol': 'SLV', 'description': 'ISHARES SILVER TRUST', 'industry': 'iShares', 'sector': 'Commodities Precious Metals'},
            {'symbol': 'SUSA', 'description': 'ISHARES MSCI USA ESG SELECT ETF', 'industry': 'Exchange Traded Fund', 'sector': 'ESG'},
            {'symbol': 'EFIV', 'description': 'SPDR S&P 500 ESG ETF', 'industry': 'Exchange Traded Fund', 'sector': 'ESG'},
            {'symbol': 'ESGV', 'description': 'VANGUARD ESG U.S. STOCK ETF', 'industry': 'Exchange Traded Fund', 'sector': 'ESG'},
            {'symbol': 'ESGU', 'description': 'ISHARES MSCI USA ESG OPTIMIZED ETF', 'industry': '', 'sector': 'ESG'},
            {'symbol': 'AFTY', 'description': 'CSOP FTSE CHINA A50 ETF', 'industry': 'CSOP Asset Management', 'sector': 'China Region'},
            {'symbol': 'MCHI', 'description': 'ISHARES MSCI CHINA', 'industry': 'iShares', 'sector': 'China Region'},
            {'symbol': 'EWH', 'description': 'ISHARES MSCI HONG KONG INDEX FU', 'industry': 'iShares', 'sector': 'China Region'},
            {'symbol': 'EEM', 'description': 'ISHARES MSCI EMERGING MARKETS', 'industry': 'iShares', 'sector': 'Diversified Emerging Mkts'},
            {'symbol': 'IEUR', 'description': 'ISHARES CORE MSCI EUROPE', 'industry': 'iShares', 'sector': 'Europe Stock'},
            {'symbol': 'VGK', 'description': 'VANGUARD FTSE EUROPE ETF', 'industry': 'Vanguard', 'sector': 'Europe Stock'},
            {'symbol': 'FLCH', 'description': 'FRANKLIN FTSE CHINA ETF', 'industry': 'Exchange Traded Fund', 'sector': 'Geographic Index'},
            {'symbol': 'EWJ', 'description': 'ISHARES MSCI JAPAN', 'industry': 'iShares', 'sector': 'Japan Stock'},
            {'symbol': 'NKY', 'description': 'MAXIS NIKKEI 225 ETF', 'industry': 'Precidian Funds LLC', 'sector': 'Japan Stock'},
            {'symbol': 'EWZ', 'description': 'ISHARES MSCI BRAZIL CAPPED', 'industry': 'iShares', 'sector': 'Latin America Stock'},
            {'symbol': 'EWC', 'description': 'ISHARES MSCI CANADA', 'industry': 'iShares', 'sector': 'Miscellaneous Region'},
            {'symbol': 'EWU', 'description': 'ISHARES MSCI UNITED KINGDOM', 'industry': 'iShares', 'sector': 'Miscellaneous Region'},
            {'symbol': 'EWI', 'description': 'ISHARES MSCI ITALY CAPPED', 'industry': 'iShares', 'sector': 'Miscellaneous Region'},
            {'symbol': 'EWP', 'description': 'ISHARES MSCI SPAIN CAPPED', 'industry': 'iShares', 'sector': 'Miscellaneous Region'},
            {'symbol': 'ACWI', 'description': 'ISHARES MSCI ACWI', 'industry': 'iShares', 'sector': 'World Stock'},
            {'symbol': 'IOO', 'description': 'ISHARES GLOBAL 100', 'industry': 'iShares', 'sector': 'World Stock'},
            {'symbol': 'GWL', 'description': 'SPDR S&P WORLD EX-US ETF', 'industry': 'SPDR State Street Global Advisors', 'sector': 'Foreign Large Blend'},
            {'symbol': 'VEU', 'description': 'VANGUARD FTSE ALL-WLD EX-US ETF', 'industry': 'Vanguard', 'sector': 'Foreign Large Blend'},
            {'symbol': 'IJH', 'description': 'ISHARES CORE S&P MID-CAP', 'industry': 'iShares', 'sector': 'Mid-Cap Blend'},
            {'symbol': 'MDY', 'description': 'SPDR S&P MIDCAP 400 ETF', 'industry': 'SPDR State Street Global Advisors', 'sector': 'Mid-Cap Blend'},
            {'symbol': 'IVOO', 'description': 'VANGUARD S&P MID-CAP 400 ETF', 'industry': 'Vanguard', 'sector': 'Mid-Cap Blend'},
            {'symbol': 'IYT', 'description': 'ISHARES TRANSPORTATION AVERAGE', 'industry': 'iShares', 'sector': 'Sector / Transportation'},
            {'symbol': 'XTN', 'description': 'SPDR S&P TRANSPORTATION ETF', 'industry': 'SPDR State Street Global Advisors', 'sector': 'Sector / Transportation'},
            {'symbol': 'XLI', 'description': 'INDUSTRIAL SELECT SECTOR SPDR ETF', 'industry': 'SPDR State Street Global Advisors', 'sector': 'Sector / Industrial'},
            {'symbol': 'XLU', 'description': 'UTILITIES SELECT SECTOR SPDR ETF', 'industry': 'SPDR State Street Global Advisors', 'sector': 'Sector / Utilities'},
            {'symbol': 'VPU', 'description': 'VANGUARD UTILITIES ETF', 'industry': 'Vanguard', 'sector': 'Sector / Utilities'},
            {'symbol': 'SPSM', 'description': 'SPDR PORTFOLIO S&P 600 SMALL CAP ETF', 'industry': 'Exchange Traded Fund', 'sector': 'Small Cap Index'},
            {'symbol': 'IJR', 'description': 'ISHARES CORE S&P SMALL-CAP', 'industry': 'iShares', 'sector': 'Small Blend'},
            {'symbol': 'VIOO', 'description': 'VANGUARD S&P SMALL-CAP 600 ETF', 'industry': 'Vanguard', 'sector': 'Small Blend'},
            {'symbol': 'QQQ', 'description': 'POWERSHARES QQQ TRUST, SERIES 1', 'industry': 'PowerShares', 'sector': 'Tech / Nasdaq'},
            {'symbol': 'ICLN', 'description': 'ISHARES GLOBAL CLEAN ENERGY', 'industry': 'iShares', 'sector': 'Thematic / Clean Energy'},
            {'symbol': 'ARKK', 'description': 'ARK INNOVATION ETF', 'industry': 'ARK ETF Trust', 'sector': 'Thematic / Innovation'},
            {'symbol': 'SPLG', 'description': 'SPDR PORTFOLIO S&P 500 ETF', 'industry': 'Exchange Traded Fund', 'sector': 'US Index'},
            {'symbol': 'SPY', 'description': 'SPDR S&P 500 ETF', 'industry': 'SPDR State Street Global Advisors', 'sector': 'US Index'},
            {'symbol': 'VOO', 'description': 'VANGUARD 500 ETF', 'industry': 'Vanguard', 'sector': 'US Index'},
            {'symbol': 'IYY', 'description': 'ISHARES DOW JONES U.S. TOTAL MA', 'industry': 'iShares', 'sector': 'US Index'},
            {'symbol': 'VTI', 'description': 'VANGUARD TOTAL STOCK MARKET ETF', 'industry': 'Vanguard', 'sector': 'US Index'},
            {'symbol': 'DIA', 'description': 'SPDR DOW JONES INDUSTRIAL AVERAGE ETF', 'industry': 'SPDR State Street Global Advisors', 'sector': 'US Index'}
        ],
        'forex': [
            {'symbol': 'EURUSD', 'description': 'EURO MEMBER COUNTRIES, EURO / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'GBPUSD', 'description': 'UNITED KINGDOM, POUNDS / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'AUDUSD', 'description': 'AUSTRALIA, DOLLARS / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'CADUSD', 'description': 'CANADA, DOLLARS / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'JPYUSD', 'description': 'JAPAN, YEN / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'CHFUSD', 'description': 'SWITZERLAND, FRANCS / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'SGDUSD', 'description': 'SINGAPORE, DOLLARS / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'HKDUSD', 'description': 'HONG KONG, DOLLARS / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'KRWUSD', 'description': 'KOREA (SOUTH), WON / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'INRUSD', 'description': 'INDIA, RUPEES / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'RUBUSD', 'description': 'RUSSIA, RUBLES / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''},
            {'symbol': 'BRLUSD', 'description': 'BRAZIL, BRAZIL REAL / UNITED STATES OF AMERICA, DOLLARS', 'industry': '', 'sector': ''}
        ],
        'futures': [
            {'symbol': 'ES', 'description': 'CONTINUOUS E-MINI S&P 500 CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'CL', 'description': 'CONTINUOUS CRUDE OIL CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'GC', 'description': 'CONTINUOUS GOLD CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'NG', 'description': 'CONTINUOUS NATURAL GAS CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'NQ', 'description': 'CONTINUOUS E-MINI NASDAQ 100 CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'TY', 'description': 'CONTINUOUS 10 YR US TREASURY NOTE CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'FV', 'description': 'CONTINUOUS 5 YR US TREASURY NOTE CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'EU', 'description': 'CONTINUOUS EURO FX CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'SI', 'description': 'CONTINUOUS SILVER CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'C', 'description': 'CONTINUOUS CORN CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'W', 'description': 'CONTINUOUS WHEAT CONTRACT', 'industry': '', 'sector': ''},
            {'symbol': 'VX', 'description': 'CONTINUOUS CBOE VOLATILITY INDEX (VIX) CONTRACT', 'industry': '', 'sector': ''}
        ]
    }
    
    assets_to_insert = []
    
    for category, assets in assets_data.items():
        print(f"Elaborazione categoria {category}: {len(assets)} asset")
        for asset in assets:
                assets_to_insert.append({
                'symbol': asset['symbol'],
                'description': asset['description'],
                'industry': asset['industry'],
                'sector': asset['sector'],
                    'asset_category': category
                })

    if not assets_to_insert:
        print("Nessun asset da inserire.")
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
    - BASE: specifici asset per ogni categoria
    - SENATOR/ADMIN: tutti gli asset di tutte le categorie
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

    # 2. Definizione degli asset specifici per utenti base
    base_assets = {
        'stocks': [
            'AAPL', 'ADBE', 'AMD', 'AMZN', 'AXP', 'BA', 'CAT', 'COIN', 'CSCO', 'DIS',
            'EBAY', 'GE', 'GOOGL', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO',
            'MCD', 'META', 'MMM', 'MSFT', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PG', 'PM',
            'PYPL', 'SHOP', 'SNAP', 'SPOT', 'TSLA', 'UBER', 'V', 'WMT', 'XOM', 'ZM'
        ],
        'etf': [
            'AGG', 'BND', 'GLD', 'SLV', 'SUSA', 'EFIV', 'ESGV', 'ESGU', 'AFTY', 'MCHI',
            'EWH', 'EEM', 'IEUR', 'VGK', 'FLCH', 'EWJ', 'NKY', 'EWZ', 'EWC', 'EWU',
            'EWI', 'EWP', 'ACWI', 'IOO', 'GWL', 'VEU', 'IJH', 'MDY', 'IVOO', 'IYT',
            'XTN', 'XLI', 'XLU', 'VPU', 'SPSM', 'IJR', 'VIOO', 'QQQ', 'ICLN', 'ARKK',
            'SPLG', 'SPY', 'VOO', 'IYY', 'VTI', 'DIA'
        ],
        'forex': ['EURUSD', 'GBPUSD', 'AUDUSD', 'CADUSD', 'JPYUSD'],
        'futures': ['ES', 'CL', 'GC', 'NG']
    }

    permissions_to_insert = []

    for user_id, role_name in users:
        asset_ids_to_assign = set()

        if role_name == 'base':
            # Per utenti base, assegna solo gli asset specifici
            for category, symbols in base_assets.items():
                if symbols:  # Se ci sono simboli specifici per questa categoria
                    placeholders = ','.join(['%s'] * len(symbols))
                    cur.execute(f"""
                        SELECT asset_id FROM assets 
                        WHERE symbol IN ({placeholders}) AND asset_category = %s
                    """, symbols + [category])
                    
                    category_assets = cur.fetchall()
                    for (asset_id,) in category_assets:
                        asset_ids_to_assign.add(asset_id)
                    
                    print(f"Utente base {user_id}: {len(category_assets)} asset {category} assegnati")

        elif role_name in ['senator', 'admin']:
            # Per senator e admin, assegna tutti gli asset
            cur.execute("SELECT asset_id FROM assets")
            all_assets = cur.fetchall()
            for (asset_id,) in all_assets:
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
    print("\nRiepilogo asset per categoria:")
    for category, symbols in base_assets.items():
        print(f"- {category.upper()}: {len(symbols)} asset per utenti base")
    
    cur.execute("SELECT asset_category, COUNT(*) FROM assets GROUP BY asset_category")
    total_counts = cur.fetchall()
    print("\nTotale asset nel database:")
    for category, count in total_counts:
        print(f"- {category.upper()}: {count} asset totali")


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
        load_assets_from_data(cur)
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