-- Create tables for test database

-- Roles table
CREATE TABLE roles (
    role_id INTEGER PRIMARY KEY,
    role_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users table
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (role_id) REFERENCES roles (role_id)
);

-- Asset access limits table
CREATE TABLE asset_access_limits (
    limit_id INTEGER PRIMARY KEY,
    role_id INTEGER NOT NULL,
    asset_type TEXT NOT NULL,
    max_symbols INTEGER NOT NULL,
    FOREIGN KEY (role_id) REFERENCES roles (role_id),
    UNIQUE (role_id, asset_type)
);

-- Realized volatility data table
CREATE TABLE realized_volatility_data (
    id INTEGER PRIMARY KEY,
    observation_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    volume REAL,
    trades INTEGER,
    open_price REAL,
    close_price REAL,
    high_price REAL,
    low_price REAL,
    rv5 REAL,
    pv REAL,
    gk REAL,
    rr5 REAL,
    rv1 REAL,
    rv5_ss REAL,
    bv1 REAL,
    bv5 REAL,
    bv5_ss REAL,
    rsp1 REAL,
    rsn1 REAL,
    rsp5 REAL,
    rsn5 REAL,
    rsp5_ss REAL,
    rsn5_ss REAL,
    medrv1 REAL,
    medrv5 REAL,
    medrv5_ss REAL,
    minrv1 REAL,
    minrv5 REAL,
    minrv5_ss REAL,
    rk REAL
);

-- Create index for performance
CREATE INDEX idx_rv_symbol ON realized_volatility_data (symbol);
CREATE INDEX idx_rv_date ON realized_volatility_data (observation_date);
CREATE INDEX idx_rv_asset_type ON realized_volatility_data (asset_type); 