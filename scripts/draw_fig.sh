#!/bin/bash

sid=2

python fig_a.py --sid=${sid};
python fig_a_ts.py --sid=${sid};

for rank in 5 10 20 30
do
  echo "rank:${rank}"
  python fig_b1.py --sid=${sid} --rank=${rank};
  python fig_b1_ts.py --sid=${sid} --rank=${rank};
  python fig_b2.py --sid=${sid} --rank=${rank};
done


