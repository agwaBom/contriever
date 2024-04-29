import json

# Temporary function (Should be removed ** Duplicated **)
def read_jsonl(path: str) -> list[dict]:
    '''
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
    '''
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]

    return data


if __name__ == "__main__":
    label_1 = '2012'
    label_2 = '2021'
    data_1 = read_jsonl('/home/khyunjin1993/dev/myRepo/temporal_alignment_rag/src/retriever/contriever/contriever/wmt_yearly_data_cut_tok/splitted/2012_dev.json')
    data_2 = read_jsonl('/home/khyunjin1993/dev/myRepo/temporal_alignment_rag/src/retriever/contriever/contriever/wmt_yearly_data_cut_tok/splitted/2021_dev.json')

    # labeling
    for data in data_1:
        data['year'] = label_1

    for data in data_2:
        data['year'] = label_2

    combined_data = data_1 + data_2

    with open('/home/khyunjin1993/dev/myRepo/temporal_alignment_rag/src/retriever/contriever/contriever/wmt_yearly_data_cut_tok/splitted/combined_2012_2021_dev.json', 'w') as f:
        for data in combined_data:
            f.write(json.dumps(data) + '\n')

    label_3 = '2016'
    data_3 = read_jsonl('/home/khyunjin1993/dev/myRepo/temporal_alignment_rag/src/retriever/contriever/contriever/wmt_yearly_data_cut_tok/splitted/2016_dev.json')

    for data in data_3:
        data['year'] = label_3
    
    combined_data = data_1 + data_2 + data_3

    with open('/home/khyunjin1993/dev/myRepo/temporal_alignment_rag/src/retriever/contriever/contriever/wmt_yearly_data_cut_tok/splitted/combined_2012_2021_2016_dev.json', 'w') as f:
        for data in combined_data:
            f.write(json.dumps(data) + '\n')
