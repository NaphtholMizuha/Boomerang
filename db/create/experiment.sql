CREATE TABLE setting (
    codename TEXT PRIMARY KEY,
    alg TEXT,
    dataset TEXT,
    model TEXT,
    n_rounds INT,
    n_epochs INT,
    split TEXT,
    n_lrn INT,
    m_lrn INT,
    atk_lrn TEXT,
    n_agg INT,
    m_agg INT,
    atk_agg TEXT
);