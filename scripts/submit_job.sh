#!/usr/bin/env bash
root="/home/ravishankar/personal_work_troja/vinparser"
cd ~/personal_work_troja/vinparser
git checkout eval_codeswitch
git pull
echo "$root/venv/bin/python $root/Runner.py --parse \
--use_cuda --train $root/data/UD_Swedish/sv-ud-train.conllu \
--dev $root/data/UD_Swedish/sv-ud-dev.conllu --test $root/data/UD_Swedish/sv-ud-test.conllu \
--config $root/config.ini" > parse.sh
if [ ! -d "results" ]; then mkdir results; fi
qsub -e "$root/results/$1" -o "$root/results/$2" -q gpu.q@dll[256] -l gpu=1,gpu_cc_min3.5=1,gpu_ram=2G parse.sh
