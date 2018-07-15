#!/usr/bin/env bash
root="/home/ravishankar/personal_work_troja/vinparser"
cd $root
git checkout eval_codeswitch_domain
mkdir -p "$root/thesis/conlls/domain"
mkdir -p "$root/thesis/models/domain"
git pull

echo "$root/venv/bin/python -u $root/Runner.py --parse --use_cuda \
--embed $root/thesis/embeds/en-hi-mapped.vec \
--train $root/thesis/data/en-hi/en-hi-domain.conllu \
--dev $root/thesis/data/en-hi/en-hi-test-domain.conllu \
--test $root/thesis/data/en-hi/en-hi-test-domain.conllu \
--save $root/thesis/models/domain/$1.vin \
--outfile $root/thesis/conlls/domain/$1.conllu \
--config $root/config.ini" > parse.sh

qsub -e "$root/thesis/results/err_$1" -o "$root/thesis/results/$1" -N "$1" -q gpu-ms.q@dll[256] -l gpu=1,gpu_cc_min3.5=1,gpu_ram=10G parse.sh
