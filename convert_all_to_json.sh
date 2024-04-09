#!bin/bash

# Convert all files to JSON format
for year in {2012..2021}; do
    python 3_convert_txt_to_json.py \
        --input wmt_yearly_data_cut_500_tok/splitted/${year}_train.txt \
        --output wmt_yearly_data_cut_500_tok/splitted/${year}_train.json
done