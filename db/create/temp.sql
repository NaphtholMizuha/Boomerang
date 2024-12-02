create or replace view feddpr2 as
select rnd, avg(loss) as avg_loss, avg(acc) as avg_acc
from feddpr
where peer_type='learner'
group by rnd;

create or replace view fedavg2 as
select rnd, avg(loss) as avg_loss, avg(acc) as avg_acc
from fedavg
group by rnd;