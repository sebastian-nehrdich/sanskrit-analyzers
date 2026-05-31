#!/bin/bash

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i python run_inf_batch.py --input-folder sanskrit-texts/subfolder$((i+1)) --mode segmentation --batch-size 500 &
done

wait
