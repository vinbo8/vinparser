#!/usr/bin/env bash
root="/home/ravishankar/personal_work_troja/vinparser"
export PYTHONPATH=$root/allennlp
cd $root
# git checkout parser_experiments
# git pull

mkdir -p "$root/experiments/conlls/embeds"

echo "$root/venv/bin/python -u $root/Runner.py --parse --use_cuda \
--train $root/experiments/data/UD_English/en-ud-train.conllu \
--dev $root/experiments/data/UD_English/en-ud-dev.conllu \
--test $root/experiments/data/UD_English/en-ud-test.conllu \
--outfile $root/experiments/conlls/$1.conllu \
--config $root/config.ini" > parse.sh
qsub -e "$root/experiments/logs/err_$1" -o "$root/experiments/logs/$1" -N "$1" -q gpu-ms.q@dll[256] -l gpu=1,gpu_cc_min3.5=1,gpu_ram=6G -p 0 parse.sh
