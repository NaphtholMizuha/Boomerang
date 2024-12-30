CREATE TABLE settings (
    id TEXT,
    dataset TEXT,
    model TEXT,
    n_rounds INT,
    n_epochs INT,
    split TEXT,
    n_lrn INT,
    m_lrn INT,
    atk_lrn TEXT,
    def_lrn TEXT,
    n_agg INT,
    m_agg INT,
    atk_agg TEXT,
    def_agg TEXT,
    PRIMARY KEY (id)
);