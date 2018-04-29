#!/usr/bin/env bash
ssh -A -t geri ssh -t sol1 "bash -s" < ./submit_job.sh $1
