#!/usr/bin/env bash
root="/home/ravishankar/personal_work_troja/vinparser"
if [[ -f "$root/results/err" ]]; then echo "error file exists: removing.. "; rm "$root/results/err"; fi
if [[ -f "$root/results/$1" ]]; then echo "$1 exists: removing.. "; rm "$root/results/$1"; fi
cd ~/personal_work_troja/vinparser
git checkout master
git pull
echo "$root/venv/bin/python $root/Runner.py --morph --use_cuda \
--train $root/data/UD_Swedish/sv-ud-train.conllu \
--dev $root/data/UD_Swedish/sv-ud-dev.conllu --test $root/data/UD_Swedish/sv-ud-test.conllu \
--config $root/config.ini" > parse.sh
if [ ! -d "results" ]; then mkdir results; fi
qsub -e "$root/results/err" -o "$root/results/$1" -q gpu.q@dll[256] -l gpu=1,gpu_cc_min3.5=1,gpu_ram=6G parse.sh
