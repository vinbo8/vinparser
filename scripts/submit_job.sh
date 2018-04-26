#!/usr/bin/env bash
root="/home/ravishankar/personal_work_troja/vinparser"
cd ~/personal_work_troja/vinparser
git checkout eval_codeswitch
git pull
touch hi_embeds.sh
echo "$root/venv/bin/python $root/Runner.py --parse \
--code_switch --use_cuda --train $root/data/codeswitch/en-hi/en-hi-train-append.conllu \
--dev $root/data/codeswitch/en-hi/en-hi-dev.conllu --test $root/data/codeswitch/en-hi/en-hi-test.conllu \
--config $root/config.ini --embed $root/data/embeds/wiki.hi.vec" > hi_embeds.sh
if [ ! -d "results" ]; then mkdir results; fi
qsub -q gpu.q@dll[256] -l gpu=1,gpu_cc_min3.5=1,gpu_ram=2G hi_embeds.sh
