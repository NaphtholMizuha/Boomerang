
-- CREATE TYPE dir_enum AS ENUM ('forward', 'backward');
CREATE TABLE score (
    id INT,
    turn INT,
    rnd INT,
    dir dir_enum,
    scores BYTEA,
    PRIMARY KEY (id, turn, rnd, dir),
    CONSTRAINT fk_config
        FOREIGN KEY (id)
        REFERENCES config (id)
        ON DELETE CASCADE
);