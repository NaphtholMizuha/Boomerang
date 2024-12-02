CREATE TABLE IF NOT EXISTS feddpr (
    rnd INT NOT NULL,
    peer_type TEXT NOT NULL,
    peer_id INT NOT NULL,
    loss FLOAT,
    acc FLOAT,
    scores FLOAT[] NOT NULL,
    CHECK (peer_type IN ('learner', 'aggregator'))
);
