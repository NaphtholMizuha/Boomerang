CREATE VIEW total AS
WITH max_per_turn AS (
    SELECT r.id, r.turn, MAX(r.acc) AS max_acc
    FROM result r
    GROUP BY r.id, r.turn
),
avg_per_id AS (
    SELECT r.id, AVG(r.max_acc) AS avg_max_acc
    FROM max_per_turn r
    GROUP BY r.id
)
SELECT c.*, ap.avg_max_acc
FROM config c JOIN avg_per_id ap ON c.id = ap.id;