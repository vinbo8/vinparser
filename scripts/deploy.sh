#!/usr/bin/env bash
ssh -A -t geri ssh -t sol1 "bash -s" < ./scripts/submit_job.sh $1
