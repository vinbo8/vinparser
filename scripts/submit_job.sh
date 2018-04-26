#!/usr/bin/env bash
cd ~/personal_work_troja/vinparser
git checkout eval_codeswitch
git pull
touch run.sh
echo "home/ravishankar/personal_work_troja/vinparser/venv/bin/python /home/ravishankar/personal_work_troja/vinparser/Runner.py --parse \
--code_switch --use_cuda --train /home/ravishankar/personal_work_troja/vinparser/data/codeswitch/en-hi/en-hi-train-append.conllu \
--dev /home/ravishankar/personal_work_troja/vinparser/data/codeswitch/en-hi/en-hi-dev.conllu --test /home/ravishankar/personal_work_troja/vinparser/data/codeswitch/en-hi/en-hi-test.conllu \
--config /home/ravishankar/personal_work_troja/vinparser/config.ini" > run.sh
if [ ! -d "results" ]; then mkdir results; fi
qsub -q gpu.q@dll[256] -l gpu=1,gpu_cc_min3.5=1,gpu_ram=2G -e "home/ravishankar/personal_work_troja/vinparser/results/$1" -o "home/ravishankar/personal_work_troja/vinparser/results/$2" run.sh
