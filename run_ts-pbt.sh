#!/bin/bash
python ./baselines/baselines/a2c/run.py --gpuid="0" >output1.txt 2>&1 &
python ./baselines/baselines/a2c/run.py --gpuid="0" >output2.txt 2>&1 &
python ./baselines/baselines/a2c/run.py --gpuid="0" >output3.txt 2>&1 &
python ./baselines/baselines/a2c/run.py --gpuid="0" >output4.txt 2>&1 &
python ./baselines/baselines/a2c/run.py --gpuid="0" >output5.txt 2>&1 &
