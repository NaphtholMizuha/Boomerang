CREATE TABLE backdoors (
    id INT,
    turn INT,
    rnd INT,
    acc FLOAT,
    PRIMARY KEY (id, turn, rnd),
    CONSTRAINT fk_params
        FOREIGN KEY (id)
        REFERENCES params (id)
        ON DELETE CASCADE
);