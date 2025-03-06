CREATE TABLE result (
    id INT,
    turn INT,
    rnd INT,
    loss FLOAT,
    acc FLOAT,
    PRIMARY KEY (id, turn, rnd),
    CONSTRAINT fk_config
        FOREIGN KEY (id)
        REFERENCES config (id)
        ON DELETE CASCADE
);