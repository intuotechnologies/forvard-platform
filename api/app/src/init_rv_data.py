# csv_to_sql_inserts.py
import csv
from datetime import datetime
import psycopg2 # Libreria per connettersi a PostgreSQL
import argparse
import os

def clean_numeric_value(value_str):
    """
    Pulisce un valore numerico stringa per l'inserimento SQL.
    Rimuove i separatori delle migliaia '.', sostituisce ',' con '.' per i decimali,
    e gestisce stringhe vuote come NULL.
    """
    if not value_str or value_str.strip() == "":
        return None # psycopg2 gestirà None come NULL SQL

    # Rimuovi i punti usati come separatori di migliaia
    cleaned_value = value_str.replace('.', '')
    # Sostituisci la virgola con il punto per i decimali
    cleaned_value = cleaned_value.replace(',', '.')
    
    try:
        # Verifica se è un numero valido (anche in notazione scientifica)
        return float(cleaned_value) # Restituisce come float, psycopg2 lo gestirà
    except ValueError:
        print(f"Attenzione: valore non convertibile in numerico '{value_str}', verrà inserito come NULL.")
        return None

def format_date_sql(date_str_ddmmyy):
    """Converte una data da DD/MM/YY a YYYY-MM-DD per SQL."""
    if not date_str_ddmmyy or date_str_ddmmyy.strip() == "":
        return None
    try:
        return datetime.strptime(date_str_ddmmyy, '%d/%m/%y').strftime('%Y-%m-%d')
    except ValueError:
        print(f"Attenzione: formato data non valido '{date_str_ddmmyy}', verrà inserito come NULL.")
        return None

def load_csv_to_postgres(
    csv_filepath='rv_data.csv', 
    table_name='realized_volatility_data',
    db_host="79.72.44.95",
    db_port="5432",
    db_name="forvard",
    db_user="admin",
    db_password="admin"
):
    """
    Legge un file CSV, si connette a PostgreSQL e inserisce i dati.
    """
    conn = None
    cur = None
    
    try:
        # Stabilisci la connessione
        print(f"Tentativo di connessione a {db_host}:{db_port}, database '{db_name}' come utente '{db_user}'...")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        cur = conn.cursor()
        print("Connessione al database PostgreSQL riuscita.")

        # Crea la tabella se non esiste
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS realized_volatility_data (
            id SERIAL PRIMARY KEY,
            observation_date DATE,
            symbol VARCHAR(20),
            asset_type VARCHAR(20),
            volume NUMERIC,
            trades INTEGER,
            open_price NUMERIC,
            close_price NUMERIC,
            high_price NUMERIC,
            low_price NUMERIC,
            pv NUMERIC,
            gk NUMERIC,
            rr5 NUMERIC,
            rv1 NUMERIC,
            rv5 NUMERIC,
            rv5_ss NUMERIC,
            bv1 NUMERIC,
            bv5 NUMERIC,
            bv5_ss NUMERIC,
            rsp1 NUMERIC,
            rsn1 NUMERIC,
            rsp5 NUMERIC,
            rsn5 NUMERIC,
            rsp5_ss NUMERIC,
            rsn5_ss NUMERIC,
            medrv1 NUMERIC,
            medrv5 NUMERIC,
            medrv5_ss NUMERIC,
            minrv1 NUMERIC,
            minrv5 NUMERIC,
            minrv5_ss NUMERIC,
            rk NUMERIC
        );
        """
        cur.execute(create_table_sql)
        conn.commit()
        print(f"Tabella {table_name} creata o già esistente.")

        with open(csv_filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter=';')
            header = next(reader) 

            db_columns = [
                "observation_date", "symbol", "asset_type", "volume", "trades",
                "open_price", "close_price", "high_price", "low_price", "pv", "gk",
                "rr5", "rv1", "rv5", "rv5_ss", "bv1", "bv5", "bv5_ss", "rsp1", "rsn1",
                "rsp5", "rsn5", "rsp5_ss", "rsn5_ss", "medrv1", "medrv5", "medrv5_ss",
                "minrv1", "minrv5", "minrv5_ss", "rk"
            ]
            
            placeholders = ', '.join(['%s'] * len(db_columns))
            sql_insert_template = f"INSERT INTO {table_name} ({', '.join(db_columns)}) VALUES ({placeholders})"

            insert_count = 0
            skipped_rows = 0
            error_rows = 0

            for i, row in enumerate(reader):
                if not any(field.strip() for field in row):
                    skipped_rows +=1
                    continue
                if len(row) != len(header):
                    print(f"Attenzione: la riga {i+2} ha un numero di colonne diverso dall'header ({len(row)} vs {len(header)}). Riga saltata: {row}")
                    skipped_rows +=1
                    continue

                try:
                    obs_date_val = format_date_sql(row[0])
                    symbol_val = row[1] if row[1] else None
                    asset_type_val = row[2] if row[2] else None
                    
                    volume_val = clean_numeric_value(row[3])
                    trades_val_raw = clean_numeric_value(row[4])
                    
                    trades_val = None
                    if trades_val_raw is not None:
                        try:
                            trades_val = int(trades_val_raw)
                        except ValueError:
                            print(f"Attenzione: 'trades' non convertibile in intero '{row[4]}', verrà inserito come NULL.")
                            trades_val = None
                    
                    open_p_val = clean_numeric_value(row[5])
                    close_p_val = clean_numeric_value(row[6])
                    high_p_val = clean_numeric_value(row[7])
                    low_p_val = clean_numeric_value(row[8])
                    pv_val = clean_numeric_value(row[9])
                    gk_val = clean_numeric_value(row[10])
                    rr5_val = clean_numeric_value(row[11])
                    rv1_val = clean_numeric_value(row[12])
                    rv5_val = clean_numeric_value(row[13])
                    rv5_ss_val = clean_numeric_value(row[14])
                    bv1_val = clean_numeric_value(row[15])
                    bv5_val = clean_numeric_value(row[16])
                    bv5_ss_val = clean_numeric_value(row[17])
                    rsp1_val = clean_numeric_value(row[18])
                    rsn1_val = clean_numeric_value(row[19])
                    rsp5_val = clean_numeric_value(row[20])
                    rsn5_val = clean_numeric_value(row[21])
                    rsp5_ss_val = clean_numeric_value(row[22])
                    rsn5_ss_val = clean_numeric_value(row[23])
                    medrv1_val = clean_numeric_value(row[24])
                    medrv5_val = clean_numeric_value(row[25])
                    medrv5_ss_val = clean_numeric_value(row[26])
                    minrv1_val = clean_numeric_value(row[27])
                    minrv5_val = clean_numeric_value(row[28])
                    minrv5_ss_val = clean_numeric_value(row[29])
                    
                    rk_val = None
                    if len(row) > 30 and row[30] is not None and row[30].strip() != "":
                         rk_val = clean_numeric_value(row[30])
                    
                    values_tuple = (
                        obs_date_val, symbol_val, asset_type_val, volume_val, trades_val,
                        open_p_val, close_p_val, high_p_val, low_p_val, pv_val, gk_val, rr5_val, rv1_val, rv5_val, rv5_ss_val,
                        bv1_val, bv5_val, bv5_ss_val, rsp1_val, rsn1_val, rsp5_val, rsn5_val, rsp5_ss_val, rsn5_ss_val,
                        medrv1_val, medrv5_val, medrv5_ss_val, minrv1_val, minrv5_val, minrv5_ss_val, rk_val
                    )
                    
                    cur.execute(sql_insert_template, values_tuple)
                    insert_count += 1
                    if insert_count % 100 == 0: 
                        print(f"Inserite {insert_count} righe...")
                        # Commit parziale per grandi file, opzionale e da valutare
                        # conn.commit() 
                        # print("Commit parziale eseguito.")


                except IndexError as ie:
                    print(f"Errore di indice durante l'elaborazione della riga {i+2}: {row}. Dettagli: {ie}")
                    error_rows += 1
                    continue
                except Exception as e:
                    print(f"Errore durante l'elaborazione della riga {i+2}: {row}")
                    print(f"Errore Python: {e}")
                    error_rows += 1
                    # Per un caricamento bulk, potresti decidere di continuare o interrompere.
                    # Per ora, continuiamo ma non facciamo il commit finale se ci sono errori.
                    # Se vuoi interrompere al primo errore, decommenta la riga sotto e il return in Exception.
                    # conn.rollback() 
                    # print("Rollback eseguito a causa di un errore.")
                    # return 
                    continue # Continua con la prossima riga

            if error_rows == 0:
                conn.commit() 
                print(f"Commit finale eseguito. Totale righe inserite: {insert_count}")
            else:
                conn.rollback()
                print(f"Rollback finale eseguito a causa di {error_rows} errori durante l'elaborazione delle righe.")
            
            print(f"Righe saltate (vuote o con numero errato di colonne): {skipped_rows}")
            print(f"Righe con errori di elaborazione (non inserite): {error_rows}")
                    
    except FileNotFoundError:
        print(f"Errore: File CSV '{csv_filepath}' non trovato.")
    except psycopg2.Error as e:
        print(f"Errore di connessione o query PostgreSQL: {e}")
        if conn:
            conn.rollback() 
    except Exception as e:
        print(f"Errore generale: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Connessione PostgreSQL chiusa.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Carica dati di volatilità dal CSV al database')
    parser.add_argument('--file', default='rv_data.csv', help='Percorso del file CSV da importare')
    parser.add_argument('--host', default='79.72.44.95', help='Host del database')
    parser.add_argument('--port', default='5432', help='Porta del database')
    parser.add_argument('--dbname', default='forvard', help='Nome del database')
    parser.add_argument('--user', default='admin', help='Utente del database')
    parser.add_argument('--password', default='admin', help='Password del database')
    
    args = parser.parse_args()
    
    # Carica le variabili d'ambiente se necessario
    db_user = os.getenv("POSTGRES_APP_USER", args.user)
    db_password = os.getenv("POSTGRES_APP_PASSWORD", args.password)
    db_host = os.getenv("POSTGRES_DB_HOST", args.host)
    db_port = os.getenv("POSTGRES_DB_PORT_INTERNAL", args.port)
    db_name = os.getenv("POSTGRES_APP_DB", args.dbname)
    
    load_csv_to_postgres(
        csv_filepath=args.file,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password
    )