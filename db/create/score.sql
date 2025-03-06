
-- CREATE TYPE dir_enum AS ENUM ('forward', 'backward');
CREATE TABLE score (
    id INT,
    turn INT,
    rnd INT,
    src INT,
    dest INT,
    dir dir_enum,
    score FLOAT,
    PRIMARY KEY (id, turn, rnd, src, dest, dir),
    CONSTRAINT fk_config
        FOREIGN KEY (id)
        REFERENCES config (id)
        ON DELETE CASCADE
);