-- Insert test data

-- Roles
INSERT INTO roles (role_id, role_name, description) 
VALUES 
    (1, 'admin', 'Administrator with full access'),
    (2, 'senator', 'Senator with extended access'),
    (3, 'base', 'Base user with limited access');

-- Users (password is hashed from: adminpass, senatorpass, userpass)
INSERT INTO users (user_id, email, password_hash, role_id) 
VALUES 
    ('11111111-1111-1111-1111-111111111111', 'admin@example.com', '$2b$12$3QnV4JD5xRjeFFMmxR3i9OstI0JWZUJ8LNRyO7TN1jLi4AsNVFM3K', 1),
    ('22222222-2222-2222-2222-222222222222', 'senator@example.com', '$2b$12$xQN6.xAMZ4YVyJl7lPYWJuiPl63uLtKX0ot5vYmR5dcP.qEi2nrwy', 2),
    ('33333333-3333-3333-3333-333333333333', 'user@example.com', '$2b$12$0lP4WVfKNTfOZAWt0p1N1uVTsGUPLhIZuP1dN6CKPsYRg1p3RZDqW', 3);

-- Asset access limits
INSERT INTO asset_access_limits (limit_id, role_id, asset_type, max_symbols) 
VALUES 
    (1, 2, 'stock', 50),     -- Senator can access 50 stocks
    (2, 2, 'crypto', 30),    -- Senator can access 30 cryptos
    (3, 2, 'forex', 20),     -- Senator can access 20 forex pairs
    (4, 3, 'stock', 10),     -- Base user can access 10 stocks
    (5, 3, 'crypto', 5),     -- Base user can access 5 cryptos
    (6, 3, 'forex', 3);      -- Base user can access 3 forex pairs

-- Sample realized volatility data
-- Stocks
INSERT INTO realized_volatility_data (observation_date, symbol, asset_type, volume, trades, open_price, close_price, high_price, low_price, rv5, pv, gk) 
VALUES 
    ('2023-01-01', 'AAPL', 'stock', 1000000, 50000, 150.0, 152.5, 153.0, 149.5, 0.15, 0.14, 0.16),
    ('2023-01-02', 'AAPL', 'stock', 1200000, 55000, 152.5, 154.0, 155.0, 152.0, 0.14, 0.13, 0.15),
    ('2023-01-03', 'AAPL', 'stock', 900000, 45000, 154.0, 153.5, 155.5, 153.0, 0.13, 0.12, 0.14),
    ('2023-01-01', 'MSFT', 'stock', 800000, 40000, 250.0, 252.0, 253.0, 249.5, 0.12, 0.11, 0.13),
    ('2023-01-02', 'MSFT', 'stock', 850000, 42000, 252.0, 253.5, 254.0, 251.5, 0.11, 0.10, 0.12),
    ('2023-01-03', 'MSFT', 'stock', 750000, 38000, 253.5, 254.0, 255.0, 253.0, 0.10, 0.09, 0.11),
    ('2023-01-01', 'GOOGL', 'stock', 600000, 30000, 120.0, 122.0, 122.5, 119.5, 0.16, 0.15, 0.17),
    ('2023-01-02', 'GOOGL', 'stock', 650000, 32000, 122.0, 123.0, 123.5, 121.5, 0.15, 0.14, 0.16),
    ('2023-01-03', 'GOOGL', 'stock', 580000, 29000, 123.0, 122.5, 123.5, 122.0, 0.14, 0.13, 0.15);

-- Crypto
INSERT INTO realized_volatility_data (observation_date, symbol, asset_type, volume, trades, open_price, close_price, high_price, low_price, rv5, pv, gk) 
VALUES 
    ('2023-01-01', 'BTC', 'crypto', 500000, 25000, 30000.0, 31000.0, 31500.0, 29800.0, 0.35, 0.33, 0.36),
    ('2023-01-02', 'BTC', 'crypto', 550000, 27000, 31000.0, 30500.0, 31200.0, 30200.0, 0.34, 0.32, 0.35),
    ('2023-01-03', 'BTC', 'crypto', 480000, 24000, 30500.0, 31200.0, 31800.0, 30400.0, 0.33, 0.31, 0.34),
    ('2023-01-01', 'ETH', 'crypto', 400000, 20000, 2000.0, 2050.0, 2080.0, 1990.0, 0.32, 0.30, 0.33),
    ('2023-01-02', 'ETH', 'crypto', 420000, 21000, 2050.0, 2100.0, 2120.0, 2040.0, 0.31, 0.29, 0.32),
    ('2023-01-03', 'ETH', 'crypto', 380000, 19000, 2100.0, 2080.0, 2110.0, 2060.0, 0.30, 0.28, 0.31);

-- Forex
INSERT INTO realized_volatility_data (observation_date, symbol, asset_type, volume, trades, open_price, close_price, high_price, low_price, rv5, pv, gk) 
VALUES 
    ('2023-01-01', 'EURUSD', 'forex', 300000, 15000, 1.1000, 1.1020, 1.1040, 1.0990, 0.08, 0.07, 0.09),
    ('2023-01-02', 'EURUSD', 'forex', 320000, 16000, 1.1020, 1.1030, 1.1050, 1.1010, 0.07, 0.06, 0.08),
    ('2023-01-03', 'EURUSD', 'forex', 290000, 14500, 1.1030, 1.1010, 1.1035, 1.1000, 0.06, 0.05, 0.07),
    ('2023-01-01', 'USDJPY', 'forex', 250000, 12500, 140.50, 140.80, 141.00, 140.40, 0.09, 0.08, 0.10),
    ('2023-01-02', 'USDJPY', 'forex', 270000, 13500, 140.80, 141.20, 141.30, 140.70, 0.08, 0.07, 0.09),
    ('2023-01-03', 'USDJPY', 'forex', 240000, 12000, 141.20, 141.00, 141.25, 140.90, 0.07, 0.06, 0.08); 