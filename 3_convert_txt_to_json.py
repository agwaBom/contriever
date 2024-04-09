'''text to json convertor'''

if __name__ == "__main__":
    import argparse
    import jsonlines

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="wmt_yearly_data/splitted/2012_train.txt")
    parser.add_argument("--output", type=str, default="wmt_yearly_data/splitted/2012_train.json")
    args = parser.parse_args()

    data = open(args.input, "r").readlines()

    outputs = []
    for line in data:
        # We do not consider the id and title fields (we only use this data to gather negative_ctxs)
        outputs.append({"id":"", "text": line, "title": ""})

    with jsonlines.open(args.output, mode='w') as fout:
        fout.write_all(outputs)
