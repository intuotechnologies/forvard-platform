-- postgres/init_db/01_schema.sql
-- Questo script verrà eseguito all'avvio del container PostgreSQL se montato in /docker-entrypoint-initdb.d

-- Creazione dei ruoli per gli utenti dell'applicazione
CREATE TABLE IF NOT EXISTS roles (
    role_id SERIAL PRIMARY KEY,
    role_name VARCHAR(50) UNIQUE NOT NULL -- Es. 'base', 'senator', 'admin'
);

-- Inserimento dei ruoli base
INSERT INTO roles (role_name) VALUES ('base'), ('senator'), ('admin')
ON CONFLICT (role_name) DO NOTHING;

-- Creazione della tabella utenti
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role_id INTEGER NOT NULL REFERENCES roles(role_id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Funzione per aggiornare updated_at
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger per aggiornare updated_at sulla tabella users
CREATE TRIGGER set_timestamp_users
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION trigger_set_timestamp();

-- Tabella per i limiti di accesso per ruolo
-- Definisce quanti asset di un certo tipo un ruolo può vedere/scaricare.
CREATE TABLE IF NOT EXISTS asset_access_limits (
    limit_id SERIAL PRIMARY KEY,
    role_id INTEGER NOT NULL REFERENCES roles(role_id),
    asset_category VARCHAR(50) NOT NULL, -- Es. 'stocks', 'futures', 'exchange_rates', 'volatility_data'
    max_items INTEGER NOT NULL,
    UNIQUE (role_id, asset_category)
);

-- Inserimento limiti per i ruoli (da "riunione_allineamento.pdf")
-- Utenti base: 40 stock, 4 futures, 4 exchange rate
INSERT INTO asset_access_limits (role_id, asset_category, max_items) VALUES
((SELECT role_id FROM roles WHERE role_name = 'base'), 'stocks', 40),
((SELECT role_id FROM roles WHERE role_name = 'base'), 'futures', 4),
((SELECT role_id FROM roles WHERE role_name = 'base'), 'exchange_rates', 4)
ON CONFLICT (role_id, asset_category) DO NOTHING;

-- Utenti senator: 80 stock, 8 futures, 8 exchange rate
INSERT INTO asset_access_limits (role_id, asset_category, max_items) VALUES
((SELECT role_id FROM roles WHERE role_name = 'senator'), 'stocks', 80),
((SELECT role_id FROM roles WHERE role_name = 'senator'), 'futures', 8),
((SELECT role_id FROM roles WHERE role_name = 'senator'), 'exchange_rates', 8)
ON CONFLICT (role_id, asset_category) DO NOTHING;

-- Per gli admin, potremmo inserire un valore molto alto o gestirlo a livello di applicazione
INSERT INTO asset_access_limits (role_id, asset_category, max_items) VALUES
((SELECT role_id FROM roles WHERE role_name = 'admin'), 'stocks', 999999),
((SELECT role_id FROM roles WHERE role_name = 'admin'), 'futures', 999999),
((SELECT role_id FROM roles WHERE role_name = 'admin'), 'exchange_rates', 999999)
ON CONFLICT (role_id, asset_category) DO NOTHING;


-- Tabella per i dati di volatilità (basata su rv_data.csv)
-- Nota: i nomi delle colonne sono stati adattati per SQL (lowercase, underscores)
-- I tipi NUMERIC sono usati per i valori finanziari per precisione.
CREATE TABLE IF NOT EXISTS realized_volatility_data (
    id SERIAL PRIMARY KEY,
    observation_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(50),
    volume BIGINT,
    trades INTEGER,
    open_price NUMERIC,
    close_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    pv NUMERIC, -- Price Variance (Potrebbe essere un nome migliore)
    gk NUMERIC, -- Garman-Klass volatility
    rr5 NUMERIC, -- Range-based Realized Volatility (5-min)
    rv1 NUMERIC, -- Realized Volatility (1-min)
    rv5 NUMERIC, -- Realized Volatility (5-min)
    rv5_ss NUMERIC, -- Realized Volatility (5-min, sub-sampled)
    bv1 NUMERIC, -- Bipower Variation (1-min)
    bv5 NUMERIC, -- Bipower Variation (5-min)
    bv5_ss NUMERIC, -- Bipower Variation (5-min, sub-sampled)
    rsp1 NUMERIC, -- Realized Semivariance (positive, 1-min)
    rsn1 NUMERIC, -- Realized Semivariance (negative, 1-min)
    rsp5 NUMERIC, -- Realized Semivariance (positive, 5-min)
    rsn5 NUMERIC, -- Realized Semivariance (negative, 5-min)
    rsp5_ss NUMERIC, -- Realized Semivariance (positive, 5-min, sub-sampled)
    rsn5_ss NUMERIC, -- Realized Semivariance (negative, 5-min, sub-sampled)
    medrv1 NUMERIC, -- Median Realized Volatility (1-min)
    medrv5 NUMERIC, -- Median Realized Volatility (5-min)
    medrv5_ss NUMERIC, -- Median Realized Volatility (5-min, sub-sampled)
    minrv1 NUMERIC, -- Min Realized Volatility (1-min)
    minrv5 NUMERIC, -- Min Realized Volatility (5-min)
    minrv5_ss NUMERIC, -- Min Realized Volatility (5-min, sub-sampled)
    rk NUMERIC, -- Realized Kernel
    UNIQUE (observation_date, symbol) -- Assicura che ogni simbolo abbia un solo record per data
);

-- Indici per migliorare le performance delle query comuni
CREATE INDEX IF NOT EXISTS idx_rv_data_date_symbol ON realized_volatility_data (observation_date, symbol);
CREATE INDEX IF NOT EXISTS idx_rv_data_symbol ON realized_volatility_data (symbol);
CREATE INDEX IF NOT EXISTS idx_rv_data_asset_type ON realized_volatility_data (asset_type);
CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_role_id ON users (role_id);

-- Commenti sulle tabelle e colonne per chiarezza
COMMENT ON TABLE roles IS 'Definisce i ruoli degli utenti (es. base, senator, admin).';
COMMENT ON TABLE users IS 'Memorizza le informazioni degli utenti, inclusa la mail e la password hashata.';
COMMENT ON TABLE asset_access_limits IS 'Definisce i limiti di accesso ai dati per ciascun ruolo e categoria di asset.';
COMMENT ON TABLE realized_volatility_data IS 'Contiene i dati di volatilità giornalieri calcolati.';
COMMENT ON COLUMN realized_volatility_data.observation_date IS 'Data di osservazione della misurazione.';
COMMENT ON COLUMN realized_volatility_data.symbol IS 'Ticker o simbolo dell''asset.';
COMMENT ON COLUMN realized_volatility_data.asset_type IS 'Tipo di asset (es. stocks, futures, exchange_rates).';

