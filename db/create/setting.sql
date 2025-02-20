CREATE TABLE params (
    id SERIAL PRIMARY KEY,
    params JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

