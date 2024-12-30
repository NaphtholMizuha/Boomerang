CREATE TABLE results (
    id TEXT,
    turn INT,
    rnd INT,
    loss FLOAT,
    acc FLOAT,
    PRIMARY KEY (id, turn, rnd)
);