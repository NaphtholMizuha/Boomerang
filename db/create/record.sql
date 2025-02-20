CREATE TABLE results (
    id INT,
    turn INT,
    rnd INT,
    loss FLOAT,
    acc FLOAT,
    PRIMARY KEY (id, turn, rnd),
    CONSTRAINT fk_params
        FOREIGN KEY (id)
        REFERENCES params (id)
        ON DELETE CASCADE
);