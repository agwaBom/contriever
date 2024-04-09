import json
def read_jsonl(path: str) -> list[dict]:
    # temp path
    data = open(path, mode='r').read().split('\n')

    # remove last line if not json
    try:
        json.loads(data[-1])
    except json.decoder.JSONDecodeError:
        print('Removing last line from jsonl file')
        print('Last line :' + data[-1])
        data = data[:-1]

    for i, line in enumerate(data):
        data[i] = json.loads(line)

    return data

if __name__ == "__main__":
    data = read_jsonl('wmt_yearly_data_cut_tok/splitted/q_2021_dev_finetuning_data.jsonl')
    print(len(data))
    
    # loop through the data
    # if key question is only one character, remove it
    for i in data:
        if len(i['question']) < 2:
            data.remove(i)
    
    print(len(data))

    # save the data
    with open('wmt_yearly_data_cut_tok/splitted/q_2021_dev_finetuning_data.jsonl', 'w') as f:
        for i in data:
            f.write(json.dumps(i) + '\n')
