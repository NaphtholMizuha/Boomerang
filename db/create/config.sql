CREATE TABLE config (
    id SERIAL PRIMARY KEY,
    mal_rate_lrn FLOAT,
    mal_rate_agg FLOAT,
    n_lrn INT,
    n_agg INT,
    model TEXT,
    data_het FLOAT,
    atk_lrn TEXT,
    def_lrn TEXT,
    atk_agg TEXT,
    def_agg TEXT,
    full_config JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

