CREATE TABLE scores (
    id INT,
    turn INT,
    rnd INT,
    src TEXT,
    dest TEXT,
    score FLOAT,
    PRIMARY KEY (id, turn, rnd, src, dest),
    CONSTRAINT fk_params
        FOREIGN KEY (id)
        REFERENCES params (id)
        ON DELETE CASCADE
);