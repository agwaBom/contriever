import os
import json
import numpy as np
import re
from tqdm import tqdm
import sys
import gc

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def print_memory_usage():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                            locals().items())), key= lambda x: -x[1]):
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

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

def remove_outliers(x, outlierConstant=2.):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList

def mine_ctx_by_threshold(input_data, negative_threshold=0.10, positive_threshold=3.0, is_positive=False, question_cnt_threshold=100):
    # Print statistics
    stat_scores = [float(x['doc_score']) for x in input_data]
    print('Mean :', str(np.mean(stat_scores)))
    print('Variance :', str(np.var(stat_scores)))
    p25 = np.percentile(stat_scores, 25)
    p75 = np.percentile(stat_scores, 75)
    print('25 percentile :', str(p25))
    print('75 percentile :', str(p75))

    # input_data = sorted(input_data, key=lambda x: float(x['query_text']))

    # Mine negative context data
    ctx_data = []
    question_cnt = 0
    tmp_question = ''
    if is_positive:
        for data in input_data:
            if float(data['doc_score']) >= p75: # and question_cnt <= question_cnt_threshold:
                ctx_data.append({
                    'question': data['query_text'],
                    'text': data['text'],
                    'doc_score': data['doc_score']
                })
                # Count the question
                if tmp_question != data['query_text']:
                    tmp_question = data['query_text']
                    question_cnt += 1

    else:
        for data in input_data:
            if float(data['doc_score']) <= p25: # and question_cnt <= question_cnt_threshold:
                ctx_data.append({
                    'question': data['query_text'],
                    'text': data['text'],
                    'doc_score': data['doc_score']
            })
                # Count the question
                if tmp_question != data['query_text']:
                    tmp_question = data['query_text']
                    question_cnt += 1 

    return ctx_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_list', type=list)
    parser.add_argument('--output', type=str, default='wmt_yearly_data_cut_tok/splitted/q_2012_dev_finetuning_data.jsonl')
    parser.add_argument('--negative_ctx_threshold', type=float, default=-10.00)
    parser.add_argument('--positive_ctx_threshold', type=float, default=10.0)
    parser.add_argument('--question_cnt', type=int, default=10000000)
    args = parser.parse_args()
 
    # Example input list
    input_path_list = ['wmt_yearly_data_cut_tok/splitted/q_2012_d_2021_dev_positive_ctx.json',
                        'wmt_yearly_data_cut_tok/splitted/q_2012_d_2012_dev_positive_ctx.json',
                        'wmt_yearly_data_cut_tok/splitted/q_2012_d_2012_dev_negative_ctx.json',
                        'wmt_yearly_data_cut_tok/splitted/q_2012_d_2021_dev_negative_ctx.json',]

    # Read input list (We remove this feature due to large memory consumption)
    # input_data = {file: read_jsonl(file) for file in input_list}

    total_ctx_data = {}
    question_list = set()
    for input_path in input_path_list:
        input_data = read_jsonl(input_path)
        year = re.findall("\d+", input_path)

        if 'negative' in input_path:
            negative_ctx_data = mine_ctx_by_threshold(input_data, negative_threshold=args.negative_ctx_threshold, positive_threshold=args.positive_ctx_threshold, is_positive=False, question_cnt_threshold=args.question_cnt)
            negative_ctx_data = sorted(negative_ctx_data, key=lambda x: x['question'])
            print_memory_usage()

            # loop over the negative_ctx_data
            for negative_ctx in tqdm(negative_ctx_data):
                # if the question is not in the total_ctx_data, we add it
                if negative_ctx['question'] not in question_list:
                    if year[0] == year[1]:
                        # add to negative_ctxs
                        total_ctx_data[negative_ctx['question']] = {'question': negative_ctx['question'], 'negative_ctxs': set([negative_ctx['text']]), 'hard_negative_ctxs': set(), 'positive_ctxs': set(), 'weak_positive_ctxs': set(), 'title':'', 'text':''}
                    elif year[0] != year[1]:
                        # add to hard_negative_ctxs
                        total_ctx_data[negative_ctx['question']] = {'question': negative_ctx['question'], 'negative_ctxs': set(), 'hard_negative_ctxs': set([negative_ctx['text']]), 'positive_ctxs': set(), 'weak_positive_ctxs': set(), 'title':'', 'text':''}
                    else:
                        print('Invalid year')
                        import IPython; IPython.embed(); exit()
                    question_list.add(negative_ctx['question'])
                else:
                    if year[0] == year[1]:
                        # add to negative_ctxs
                        try:
                            total_ctx_data[negative_ctx['question']]['negative_ctxs'].add(negative_ctx['text'])
                        except:
                            print('Error')
                            import IPython; IPython.embed(); exit()
                    elif year[0] != year[1]:
                        # add to hard_negative_ctxs
                        try:
                            total_ctx_data[negative_ctx['question']]['hard_negative_ctxs'].add(negative_ctx['text'])
                        except:
                            print('Error')
                            import IPython; IPython.embed(); exit()
                    else:
                        print('Invalid year')
                        import IPython; IPython.embed(); exit()
            del negative_ctx_data[:]; del negative_ctx_data

        elif 'positive' in input_path:
            positive_ctx_data = mine_ctx_by_threshold(input_data, negative_threshold=args.negative_ctx_threshold, positive_threshold=args.positive_ctx_threshold, is_positive=True, question_cnt_threshold=args.question_cnt)
            positive_ctx_data = sorted(positive_ctx_data, key=lambda x: x['question'])

            # loop over the positive_ctx_data
            for positive_ctx in tqdm(positive_ctx_data):
                # if the question is not in the total_ctx_data, we add it
                if positive_ctx['question'] not in question_list:
                    if year[0] == year[1]:
                        # add to negative_ctxs
                        total_ctx_data[positive_ctx['question']] = {'question': positive_ctx['question'], 'negative_ctxs': set(), 'hard_negative_ctxs': set(), 'positive_ctxs': set([positive_ctx['text']]), 'weak_positive_ctxs': set(), 'title':'', 'text':''}
                    elif year[0] != year[1]:
                        # add to hard_negative_ctxs
                        total_ctx_data[positive_ctx['question']] = {'question': positive_ctx['question'], 'negative_ctxs': set(), 'hard_negative_ctxs': set(), 'positive_ctxs': set(),'weak_positive_ctxs': set(positive_ctx['text']), 'title':'', 'text':''}
                    question_list.add(positive_ctx['question'])
                else:
                    if year[0] == year[1]:
                        # add to positive_ctxs
                        total_ctx_data[positive_ctx['question']]['positive_ctxs'].add(positive_ctx['text'])
                    elif year[0] != year[1]:
                        # add to weak_positive_ctxs
                        total_ctx_data[positive_ctx['question']]['weak_positive_ctxs'].add(positive_ctx['text'])
            del positive_ctx_data[:]; del positive_ctx_data

        else:
            print('Invalid input path')
            exit()
        
        # Free the input_data on memory
        del input_data[:]; del input_data
        gc.collect()

    # extract the data that only contains all the content (positive, negative, hard_negative)
    ctx_data = []
    len_list = []

    # [i for i in len_list if i[2] > 0]
    for ctx_key in total_ctx_data.keys():
        ctx = total_ctx_data[ctx_key]
        len_list.append([len(ctx['negative_ctxs']), len(ctx['hard_negative_ctxs']), len(ctx['positive_ctxs']), len(ctx['weak_positive_ctxs'])])
        if len(ctx['negative_ctxs']) > 0 and len(ctx['hard_negative_ctxs']) > 0 and len(ctx['positive_ctxs']) > 0 and len(ctx['weak_positive_ctxs']) > 0:
            ctx_data.append(ctx)

    with open(args.output+'.total', 'w') as f:
        for ctx in total_ctx_data.keys():
            ctx = total_ctx_data[ctx]
            f.write(json.dumps(ctx, default=set_default) + '\n')

    # Save the ctx_data
    with open(args.output, 'w') as f:
        for ctx in ctx_data:
            f.write(json.dumps(ctx, default=set_default) + '\n')

    # save len_list
    with open(args.output+'.len', 'w') as f:
        f.write('negative_ctxs\thard_negative_ctxs\tpositive_ctxs\tweak_positive_ctxs\n')
        for l in len_list:
            f.write(f'{l[0]}\t{l[1]}\t{l[2]}\t{l[3]}\n')

