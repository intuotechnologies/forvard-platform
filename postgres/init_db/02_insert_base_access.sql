-- 1. Aggiungiamo i permessi per gli stock specifici (evitando duplicati)
INSERT INTO asset_permissions (user_id, asset_id)
SELECT '9cfd3e18-6224-4030-acc0-52a87b0b00c9', asset_id
FROM assets 
WHERE symbol IN (
    'AAPL', 'ADBE', 'AMD', 'AMZN', 'AXP', 'BA', 'CAT', 'COIN', 'CSCO', 'DIS',
    'EBAY', 'GE', 'GOOGL', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO',
    'MCD', 'META', 'MMM', 'MSFT', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PG', 'PM',
    'PYPL', 'SHOP', 'SNAP', 'SPOT', 'TSLA', 'UBER', 'V', 'WMT', 'XOM', 'ZM'
)
AND asset_category = 'stocks'
AND asset_id NOT IN (
    SELECT asset_id FROM asset_permissions 
    WHERE user_id = '9cfd3e18-6224-4030-acc0-52a87b0b00c9'
);

-- 2. Aggiungiamo i permessi per tutti gli ETF
INSERT INTO asset_permissions (user_id, asset_id)
SELECT '9cfd3e18-6224-4030-acc0-52a87b0b00c9', asset_id
FROM assets 
WHERE asset_category = 'etf'
AND asset_id NOT IN (
    SELECT asset_id FROM asset_permissions 
    WHERE user_id = '9cfd3e18-6224-4030-acc0-52a87b0b00c9'
);

-- 3. Aggiungiamo i permessi per i forex specifici
INSERT INTO asset_permissions (user_id, asset_id)
SELECT '9cfd3e18-6224-4030-acc0-52a87b0b00c9', asset_id
FROM assets 
WHERE symbol IN ('EURUSD', 'GBPUSD', 'AUDUSD', 'CADUSD', 'JPYUSD')
AND asset_category = 'forex'
AND asset_id NOT IN (
    SELECT asset_id FROM asset_permissions 
    WHERE user_id = '9cfd3e18-6224-4030-acc0-52a87b0b00c9'
);

-- 4. Aggiungiamo i permessi per i futures specifici
INSERT INTO asset_permissions (user_id, asset_id)
SELECT '9cfd3e18-6224-4030-acc0-52a87b0b00c9', asset_id
FROM assets 
WHERE symbol IN ('ES', 'CL', 'GC', 'NG')
AND asset_category = 'futures'
AND asset_id NOT IN (
    SELECT asset_id FROM asset_permissions 
    WHERE user_id = '9cfd3e18-6224-4030-acc0-52a87b0b00c9'
);