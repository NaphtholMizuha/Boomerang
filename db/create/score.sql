CREATE TABLE scores (
    id TEXT,
    turn INT,
    rnd INT,
    src TEXT,
    dest TEXT,
    score FLOAT,
    PRIMARY KEY (id, turn, rnd, src, dest)
);