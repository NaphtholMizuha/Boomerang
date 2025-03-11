CREATE TABLE config (
    id SERIAL PRIMARY KEY,
    mal_rate_c FLOAT,
    mal_rate_s FLOAT,
    n_c INT,
    n_s INT,
    model TEXT,
    data_het FLOAT,
    atk_c TEXT,
    def_c TEXT,
    atk_s TEXT,
    def_s TEXT,
    full_config JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

