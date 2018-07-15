#!/usr/bin/env bash
root="/home/ravishankar/personal_work_troja/vinparser"
cd $root
git checkout eval_codeswitch_domain
git pull
echo "$root/venv/bin/python -u $root/Runner.py --parse --use_cuda \
--embed $root/thesis/embeds/en-hi-mapped.vec \
--train $root/thesis/data/en-hi/en-hi-train-domain.conllu \
--dev $root/thesis/data/en-hi/en-hi-test-domain.conllu \
--test $root/thesis/data/en-hi/en-hi-test-domain.conllu \
--save $root/thesis/models/$1.vin \
--outfile $root/thesis/conlls/$1.conllu \
--config $root/config.ini" > parse.sh
qsub -e "$root/thesis/results/err" -o "$root/thesis/results/$1" -N "$1" -q gpu-ms.q@dll[256] -l gpu=1,gpu_cc_min3.5=1,gpu_ram=10G parse.sh
