#!/usr/bin/env bash
root="/home/ravishankar/personal_work_troja/vinparser"
cd $root
git checkout eval_codeswitch
git pull

mkdir -p "$root/thesis/conlls/embeds"

echo "$root/venv/bin/python -u $root/Runner.py --parse --use_cuda --chars \
--embed $root/thesis/embeds/en-hi-append.vec \
--train $root/thesis/data/en-hi/en-hi-train-append.conllu \
--dev $root/thesis/data/en-hi/en-hi-test.conllu \
--test $root/thesis/data/en-hi/en-hi-test.conllu \
--save $root/thesis/models/$1.vin \
--outfile $root/thesis/conlls/embeds/$1.conllu \
--config $root/config.ini" > parse.sh
qsub -e "$root/thesis/results/err_$1" -o "$root/thesis/results/$1" -N "$1" -q gpu-ms.q@dll[256] -l gpu=1,gpu_cc_min3.5=1,gpu_ram=10G -p 0 parse.sh
