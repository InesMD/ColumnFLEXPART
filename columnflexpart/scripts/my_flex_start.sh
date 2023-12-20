#!/bin/bash
cd /home/b/b382105/ColumnFLEXPART/columnflexpart/scripts/
for i in {0..2}
do
	python flexpart_safe_start.py ./configs/safe_start.yaml $i &
done
wait
