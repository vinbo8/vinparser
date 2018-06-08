#!/usr/bin/env bash
root="/home/ravishankar/personal_work_troja/vinparser"
cd $root
git checkout eval_codeswitch
git pull
echo "$root/venv/bin/python -u $root/Runner.py --parse --use_cuda \
--train $root/thesis/data/en-hi/en-ud-append.conllu \
--dev $root/thesis/data/en-hi/en-ud-dev.conllu \
--test $root/thesis/data/en-hi/en-ud-test.conllu \
--embed $root/thesis/embeds/hi_en_append.vec --config $root/config.ini" > parse.sh
qsub -e "$root/thesis/results/err" -o "$root/thesis/results/$1" -N "$1" -q gpu-ms.q@dll[256] -l gpu=1,gpu_cc_min3.5=1,gpu_ram=6G parse.sh
