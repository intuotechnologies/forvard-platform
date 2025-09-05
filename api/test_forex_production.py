#!/usr/bin/env python3
"""
Test per il download dei dati forex su API di produzione
URL: http://volare.unime.it:8443
"""
import requests
import json

def test_forex_download_production():
    base_url = 'http://volare.unime.it:8443'
    
    print('=== TEST DOWNLOAD FOREX - API PRODUZIONE ===')
    print(f'URL Base: {base_url}')
    
    # Login utente base
    print('\n--- Login utente base ---')
    login_data = {'username': 'base@example.com', 'password': 'basepass'}
    
    try:
        response = requests.post(f'{base_url}/auth/token', data=login_data, timeout=10)
        print(f'Status Code Login: {response.status_code}')
        
        if response.status_code != 200:
            print(f'❌ Login fallito: {response.status_code}')
            print('Response:', response.text[:300])
            return
        
        token = response.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}
        print('✅ Login riuscito!')
        
    except requests.exceptions.RequestException as e:
        print(f'❌ Errore di connessione durante login: {e}')
        return
    
    # Test 1: Visualizza dati forex (senza download)
    print('\n--- Test 1: Visualizzazione dati forex ---')
    try:
        response = requests.get(f'{base_url}/financial-data?asset_type=forex&limit=50', 
                              headers=headers, timeout=15)
        
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            print(f'✅ Dati forex recuperati: {len(data["data"])} records')
            
            # Mostra simboli unici
            unique_symbols = set(item['symbol'] for item in data['data'])
            print(f'Simboli forex disponibili: {sorted(unique_symbols)}')
            print(f'Numero simboli unici: {len(unique_symbols)}')
            
            # Mostra primi 3 records
            print('\nPrimi 3 records:')
            for i, record in enumerate(data['data'][:3], 1):
                # Mostra tutti i campi disponibili del primo record
                if i == 1:
                    print(f'Campi disponibili: {list(record.keys())}')
                
                # Mostra record con alcuni campi principali
                date = record.get('observation_date', 'N/A')
                symbol = record.get('symbol', 'N/A')
                asset_type = record.get('asset_type', 'N/A')
                rv1 = record.get('rv1', 'N/A')
                print(f'{i}. {symbol} ({asset_type}) - {date} - RV1: {rv1}')
                
        else:
            print(f'❌ Visualizzazione fallita: {response.status_code}')
            try:
                error_data = response.json()
                print('Errore:', json.dumps(error_data, indent=2))
            except:
                print('Errore (raw):', response.text[:300])
                
    except requests.exceptions.RequestException as e:
        print(f'❌ Errore di connessione durante visualizzazione: {e}')
    
    # Test 2: Download CSV forex
    print('\n--- Test 2: Download CSV forex ---')
    try:
        response = requests.get(f'{base_url}/financial-data/download?asset_type=forex&limit=100', 
                              headers=headers, timeout=30)
        
        print(f'Status Code: {response.status_code}')
        print(f'Content-Type: {response.headers.get("content-type", "N/A")}')
        print(f'Content-Length: {response.headers.get("content-length", "N/A")} bytes')
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            
            if 'text/csv' in content_type:
                content = response.text
                lines = content.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                
                print(f'\n✅ DOWNLOAD FOREX RIUSCITO!')
                print(f'Formato: CSV')
                print(f'Righe totali: {len(non_empty_lines)}')
                
                # Mostra header e prime righe
                print('\nHeader CSV:')
                if non_empty_lines:
                    print(non_empty_lines[0])
                
                print('\nPrime 5 righe dati:')
                for i, line in enumerate(non_empty_lines[1:6], 1):
                    if line.strip():
                        print(f'{i}: {line}')
                        
            elif 'application/json' in content_type:
                try:
                    data = response.json()
                    print(f'\n✅ DOWNLOAD RIUSCITO!')
                    print(f'Formato: JSON')
                    print(f'Records: {len(data) if isinstance(data, list) else "N/A"}')
                    
                    if isinstance(data, list) and data:
                        print(f'Primo record: {data[0]}')
                    elif isinstance(data, dict):
                        print(f'Struttura: {list(data.keys())}')
                        
                except json.JSONDecodeError:
                    print('✅ Download riuscito ma contenuto non è JSON valido')
                    
            else:
                print(f'✅ Download riuscito!')
                print(f'Content Length: {len(response.content)} bytes')
                print(f'Content preview: {str(response.content[:200])}...')
                
        else:
            print(f'❌ Download fallito: {response.status_code}')
            try:
                error_data = response.json()
                print('Errore:', json.dumps(error_data, indent=2))
            except:
                print('Errore (raw):', response.text[:300])
                
    except requests.exceptions.RequestException as e:
        print(f'❌ Errore di connessione durante download: {e}')
    
    # Test 3: Verifica permessi (test con utente admin se disponibile)
    print('\n--- Test 3: Test utente admin (opzionale) ---')
    try:
        admin_login_data = {'username': 'admin@example.com', 'password': 'adminpass'}
        response = requests.post(f'{base_url}/auth/token', data=admin_login_data, timeout=10)
        
        if response.status_code == 200:
            admin_token = response.json()['access_token']
            admin_headers = {'Authorization': f'Bearer {admin_token}'}
            
            response = requests.get(f'{base_url}/financial-data?asset_type=forex&limit=50', 
                                  headers=admin_headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                admin_symbols = set(item['symbol'] for item in data['data'])
                print(f'✅ Admin login riuscito')
                print(f'Simboli forex per admin: {len(admin_symbols)} simboli')
                print(f'Simboli: {sorted(admin_symbols)}')
            else:
                print(f'❌ Admin data request failed: {response.status_code}')
        else:
            print('ℹ️  Admin login non disponibile o credenziali diverse')
            
    except requests.exceptions.RequestException as e:
        print(f'ℹ️  Test admin non completato: {e}')

if __name__ == "__main__":
    test_forex_download_production()
