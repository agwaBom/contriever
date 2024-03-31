#!/bin/bash

for year in {2012..2021}; do
    bash ./data_scripts/tokenization_script.sh wmt_yearly_data/splitted/${year}_train.txt
done

for year in {2012..2021}; do
    bash ./data_scripts/tokenization_script.sh wmt_yearly_data/splitted/${year}_dev.txt
done