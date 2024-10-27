import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
from transformers import BertConfig, BertModel
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_local_contriever(path: str, train_strategy: str):
    # train_strategy = "moco"
    # path = 'checkpoint/tmp/contriever_inbatch/checkpoint/step-500/checkpoint.pth'
    model = torch.load(path)
    model = model['model']
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

    # iterate model keys and get only encoder_q keys
    encoder = {}
    if train_strategy == "moco":
        for key in model.keys():
            # in MoCo, we only use encoder_q
            if 'encoder_q' in key:
                # remove encoder_q. from key
                encoder[key.removeprefix("encoder_q.")] = model[key]
    elif train_strategy == "inbatch":
        for key in model.keys():
            if 'encoder' in key:
                # remove encoder. from key
                encoder[key.removeprefix("encoder.")] = model[key]
    # can be this simple
    # contriever = model.get_encoder()

    config = BertConfig.from_pretrained("facebook/contriever")
    contriever = BertModel(config)

    if encoder == {}:
        encoder = model

    # Load the weights from the pre-trained model we set strict to False to ignore the pooler layer
    contriever.load_state_dict(encoder, strict=False)

    # Check if the partial weights are equal
    assert torch.all(contriever.embeddings.word_embeddings.weight.detach().cpu() == encoder['embeddings.word_embeddings.weight'].cpu()), "weights are not equal"

    return contriever, tokenizer


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def read_jsonl(path: str) -> list[dict]:
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip() != '']
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', nargs='+', type=str, default=['/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/SituatedQA/data/qa_data/temp.train.jsonl'])
    parser.add_argument('--model_path', type=str, required=False, default='/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/things_to_send/run_situatedqa/checkpoint/20171220-20181220_margin_1_bz_128_2_10_step_const_lr_0_8_0_2/checkpoint/step-8000/checkpoint.pth')
    #parser.add_argument('--model_path', type=str, required=False, default='/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/things_to_send/run_situatedqa/checkpoint/20181220-20211220_margin_1_bz_128_2_10_step_const_lr_0_8_0_2/checkpoint/step-20000/checkpoint.pth')

    parser.add_argument('--model_alias', type=str, required=False, default='2018')
    args = parser.parse_args()

    # load base contriever
    # base_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
    base_model = AutoModel.from_pretrained("facebook/contriever")
    if args.model_path[-4:] == ".pth":
        temp_model, _ = load_local_contriever(args.model_path, "moco")
    else:
        temp_model = AutoModel.from_pretrained(args.model_path)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    base_model = base_model.to(device=device)
    temp_model = temp_model.to(device=device)

    # load data
    print("Loading data...")
    data_list = []
    for path in args.input_path:
        data = read_jsonl(path)
        data_list.append(data)
    
    '''
    {'question': 'where will the next summer and winter olympics be held',
    'id': 2098168902147822379,
    'edited_question': 'where will the next summer and winter olympics be held as of 2021',
    'date': '2021',
    'date_type': 'sampled_year',
    'answer': ['Japan and China'],
    'any_answer': ['Brazil and S. Korea', 'Japan and China']}
    '''

    # extract embeddings
    print("Extracting embeddings...")
    for data in data_list:
        for i in tqdm(range(len(data))):
            text = f"Question: {data[i]['edited_question']} Answer: {' '.join(data[i]['any_answer'])}"
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = base_model(inputs["input_ids"].to(device=device), inputs['attention_mask'].to(device=device), output_hidden_states=True)
                # [1, 768]
                base_embedding = mean_pooling(outputs['hidden_states'][-1], inputs['attention_mask'].to(device=device))

            with torch.no_grad():
                outputs = temp_model(inputs["input_ids"].to(device=device), inputs['attention_mask'].to(device=device), output_hidden_states=True)
                # [1, 768]
                temp_embedding = mean_pooling(outputs['hidden_states'][-1], inputs['attention_mask'].to(device=device))

            # calculate cosine distance
            cosine_distance = torch.nn.functional.cosine_similarity(base_embedding, temp_embedding, dim=1)
            data[i]['cosine_distance'] = cosine_distance.item()
            print(data[i]['cosine_distance'])

    # save data
    print("Saving data...")
    for i, path in enumerate(args.input_path):
        with open(f"{path[:-6]}.{args.model_alias}.jsonl", 'w') as f:
            for data in data_list[i]:
                f.write(json.dumps(data) + "\n")
