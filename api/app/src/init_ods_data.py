# ods_to_sql_inserts.py
import pandas as pd
from datetime import datetime
import psycopg2 # Libreria per connettersi a PostgreSQL
import argparse
import os

def clean_numeric_value(value_str):
    """
    Pulisce un valore numerico stringa per l'inserimento SQL.
    Gestisce separatori delle migliaia '.', decimali ',', notazione scientifica,
    e stringhe vuote o placeholder come NULL.
    """
    if not value_str or value_str.strip() == "" or value_str.strip().lower() == '########':
        return None 

    original_value_str = value_str 
    value_str = value_str.strip()

    if 'e' in value_str.lower():
        value_str = value_str.replace(',', '.')
        try:
            return float(value_str)
        except ValueError:
            print(f"Attenzione: valore scientifico non convertibile '{original_value_str}', verrà inserito come NULL.")
            return None

    if ',' in value_str:
        value_str = value_str.replace('.', '')  
        value_str = value_str.replace(',', '.')  
    elif '.' in value_str:
        num_dots = value_str.count('.')
        if num_dots > 1:
            value_str = value_str.replace('.', '') 
        else: 
              pass 
    try:
        return float(value_str)
    except ValueError:
        print(f"Attenzione: valore non convertibile in numerico '{original_value_str}' (processato come '{value_str}'), verrà inserito come NULL.")
        return None

def format_date_sql(date_str_ddmmyy):
    """Converte una data da DD/MM/YY a YYYY-MM-DD per SQL."""
    if not date_str_ddmmyy or date_str_ddmmyy.strip() == "":
        return None
    try:
        # If it's already a datetime object from pandas, format it
        if isinstance(date_str_ddmmyy, datetime):
            return date_str_ddmmyy.strftime('%Y-%m-%d')
        return datetime.strptime(date_str_ddmmyy, '%d/%m/%y').strftime('%Y-%m-%d')
    except ValueError:
        # Try parsing with a more general date parser if a string was passed
        try:
            # Pandas might have read it as a string like 'YYYY-MM-DD HH:MM:SS'
            dt_obj = pd.to_datetime(date_str_ddmmyy).to_pydatetime()
            return dt_obj.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            print(f"Attenzione: formato data non valido '{date_str_ddmmyy}', verrà inserito come NULL.")
            return None


def _to_str_for_cleaners(val, col_name_for_debug=""):
    """Helper to convert pandas cell values to string for processing, handling dates."""
    if pd.isna(val) or (isinstance(val, str) and not val.strip()):
        return "" 
    # If pandas parsed a date and format_date_sql expects dd/mm/yy string for its strptime
    if isinstance(val, datetime):
        try:
            return val.strftime('%d/%m/%y') 
        except ValueError:
            print(f"Attenzione: data da pandas non formattabile '{val}' per colonna '{col_name_for_debug}', trattata come stringa.")
            # Fallback to string representation if strftime fails for some reason
            return str(val)
    return str(val)

def load_ods_to_postgres(
    ods_filepath='data.ods', 
    table_name='realized_volatility_data',
    db_host="79.72.44.95",
    db_port="5432",
    db_name="forvard",
    db_user="admin",
    db_password="admin"
):
    """
    Legge un file ODS, si connette a PostgreSQL e inserisce i dati.
    """
    conn = None
    cur = None
    
    try:
        print(f"Tentativo di lettura del file ODS: {ods_filepath}")
        try:
            df = pd.read_excel(ods_filepath, engine='odf', header=0)
        except ImportError:
            print("Errore: La libreria 'odfpy' è necessaria per leggere i file ODS.")
            print("Installala con: pip install odfpy")
            return
        except FileNotFoundError:
            print(f"Errore: File ODS '{ods_filepath}' non trovato.")
            return
        except Exception as e:
            print(f"Errore durante la lettura del file ODS '{ods_filepath}': {e}")
            return
            
        print(f"File ODS '{ods_filepath}' letto con successo. Righe lette: {len(df)}")
        if df.empty:
            print("Il file ODS è vuoto o non contiene dati nelle colonne attese.")
            return

        print(f"Tentativo di connessione a {db_host}:{db_port}, database '{db_name}' come utente '{db_user}'...")
        conn = psycopg2.connect(
            host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password
        )
        cur = conn.cursor()
        print("Connessione al database PostgreSQL riuscita.")

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS realized_volatility_data (
            id SERIAL PRIMARY KEY, observation_date DATE, symbol VARCHAR(20), asset_type VARCHAR(20),
            volume NUMERIC, trades INTEGER, open_price NUMERIC, close_price NUMERIC,
            high_price NUMERIC, low_price NUMERIC, pv NUMERIC, gk NUMERIC, rr5 NUMERIC,
            rv1 NUMERIC, rv5 NUMERIC, rv5_ss NUMERIC, bv1 NUMERIC, bv5 NUMERIC, bv5_ss NUMERIC,
            rsp1 NUMERIC, rsn1 NUMERIC, rsp5 NUMERIC, rsn5 NUMERIC, rsp5_ss NUMERIC, rsn5_ss NUMERIC,
            medrv1 NUMERIC, medrv5 NUMERIC, medrv5_ss NUMERIC, minrv1 NUMERIC, minrv5 NUMERIC,
            minrv5_ss NUMERIC, rk NUMERIC
        );"""
        cur.execute(create_table_sql)
        conn.commit()
        print(f"Tabella {table_name} creata o già esistente.")

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
        
        expected_min_cols = len(db_columns) -1 # rk column is optional

        for i, row_from_ods in enumerate(df.itertuples(index=False, name=None)): 
            # row_from_ods is likely a tuple with one element: the comma-separated string
            # e.g., ('2024-03-01,GE,stocks,6046871,...',)

            if not row_from_ods or not row_from_ods[0] or not isinstance(row_from_ods[0], str):
                print(f"Attenzione: riga ODS {i+1} ha un formato inatteso, è vuota o non è una stringa. Riga saltata: {str(row_from_ods)[:100]}")
                skipped_rows +=1
                continue

            # Split the single string into a list of fields
            row_tuple = row_from_ods[0].split(',')

            raw_row_for_log = tuple(str(v)[:50] for v in row_tuple) # For logging brevity

            if not any(str(field).strip() for field in row_tuple if not pd.isna(field)):
                skipped_rows +=1
                continue
            
            if len(row_tuple) < expected_min_cols:
                print(f"Attenzione: riga {i+1} ha {len(row_tuple)} colonne, attese almeno {expected_min_cols}. Riga saltata: {raw_row_for_log}")
                skipped_rows +=1
                continue

            try:
                # Get string representations for processing
                s = lambda idx, name="": _to_str_for_cleaners(row_tuple[idx] if len(row_tuple) > idx else "", name)

                obs_date_val = format_date_sql(s(0, "obs_date"))
                symbol_val = s(1, "symbol") if s(1, "symbol") else None
                asset_type_val = s(2, "asset_type") if s(2, "asset_type") else None
                
                volume_val = clean_numeric_value(s(3, "volume"))
                trades_val_raw_str = s(4, "trades")
                trades_val_cleaned = clean_numeric_value(trades_val_raw_str)
                
                trades_val = None
                if trades_val_cleaned is not None:
                    try:
                        trades_val = int(trades_val_cleaned) # clean_numeric_value returns float
                    except ValueError:
                        print(f"Attenzione: 'trades' ('{trades_val_raw_str}' -> {trades_val_cleaned}) non convertibile in intero, verrà inserito come NULL.")
                
                open_p_val = clean_numeric_value(s(5, "open_p"))
                close_p_val = clean_numeric_value(s(6, "close_p"))
                high_p_val = clean_numeric_value(s(7, "high_p"))
                low_p_val = clean_numeric_value(s(8, "low_p"))
                pv_val = clean_numeric_value(s(9, "pv"))
                gk_val = clean_numeric_value(s(10, "gk"))
                rr5_val = clean_numeric_value(s(11, "rr5"))
                rv1_val = clean_numeric_value(s(12, "rv1"))
                rv5_val = clean_numeric_value(s(13, "rv5"))
                rv5_ss_val = clean_numeric_value(s(14, "rv5_ss"))
                bv1_val = clean_numeric_value(s(15, "bv1"))
                bv5_val = clean_numeric_value(s(16, "bv5"))
                bv5_ss_val = clean_numeric_value(s(17, "bv5_ss"))
                rsp1_val = clean_numeric_value(s(18, "rsp1"))
                rsn1_val = clean_numeric_value(s(19, "rsn1"))
                rsp5_val = clean_numeric_value(s(20, "rsp5"))
                rsn5_val = clean_numeric_value(s(21, "rsn5"))
                rsp5_ss_val = clean_numeric_value(s(22, "rsp5_ss"))
                rsn5_ss_val = clean_numeric_value(s(23, "rsn5_ss"))
                medrv1_val = clean_numeric_value(s(24, "medrv1"))
                medrv5_val = clean_numeric_value(s(25, "medrv5"))
                medrv5_ss_val = clean_numeric_value(s(26, "medrv5_ss"))
                minrv1_val = clean_numeric_value(s(27, "minrv1"))
                minrv5_val = clean_numeric_value(s(28, "minrv5"))
                minrv5_ss_val = clean_numeric_value(s(29, "minrv5_ss"))
                
                rk_val = None
                if len(row_tuple) > 30:
                     rk_val = clean_numeric_value(s(30, "rk"))
                
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
                    # conn.commit() # Consider partial commits for very large files

            except IndexError as ie: # Should be caught by len(row_tuple) check, but as a safeguard
                print(f"Errore di indice (ODS) durante l'elaborazione della riga {i+1}: {raw_row_for_log}. Dettagli: {ie}")
                error_rows += 1
            except Exception as e:
                print(f"Errore durante l'elaborazione della riga {i+1} (ODS): {raw_row_for_log}")
                print(f"Errore Python: {e}")
                error_rows += 1
                # conn.rollback() # Rollback this row? Or collect errors and rollback at end.
                # return 

        if error_rows == 0:
            conn.commit() 
            print(f"Commit finale eseguito. Totale righe inserite: {insert_count}")
        else:
            conn.rollback()
            print(f"Rollback finale eseguito a causa di {error_rows} errori durante l'elaborazione delle righe.")
        
        print(f"Righe saltate (vuote o con numero errato di colonne): {skipped_rows}")
        print(f"Righe con errori di elaborazione (non inserite): {error_rows}")
                
    except psycopg2.Error as e:
        print(f"Errore di connessione o query PostgreSQL: {e}")
        if conn: conn.rollback() 
    except Exception as e:
        print(f"Errore generale: {e}")
        if conn: conn.rollback()
    finally:
        if cur: cur.close()
        if conn:
            conn.close()
            print("Connessione PostgreSQL chiusa.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Carica dati di volatilità da un file ODS al database PostgreSQL')
    parser.add_argument('--file', default='data.ods', help='Percorso del file ODS da importare (default: data.ods)')
    parser.add_argument('--host', default='79.72.44.95', help='Host del database')
    parser.add_argument('--port', default='5432', help='Porta del database')
    parser.add_argument('--dbname', default='forvard', help='Nome del database')
    parser.add_argument('--user', default='admin', help='Utente del database')
    parser.add_argument('--password', default='admin', help='Password del database')
    
    args = parser.parse_args()
    
    db_user = os.getenv("POSTGRES_APP_USER", args.user)
    db_password = os.getenv("POSTGRES_APP_PASSWORD", args.password)
    db_host = os.getenv("POSTGRES_DB_HOST", args.host)
    db_port = os.getenv("POSTGRES_DB_PORT_INTERNAL", args.port)
    db_name = os.getenv("POSTGRES_APP_DB", args.dbname)
    
    load_ods_to_postgres(
        ods_filepath=args.file,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password
    ) 