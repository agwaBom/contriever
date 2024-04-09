from datasets import load_dataset
from transformers import BertTokenizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

TOKENIZER = BertTokenizer.from_pretrained("facebook/contriever")
WMT_YEAR_DATA = load_dataset("KaiNylund/WMT-year-splits")


def process_split(split):
    wmt_year_data = WMT_YEAR_DATA[split]
    tok_split = False

    split_data = []
    for t in tqdm(wmt_year_data['text']):
        token_seq = TOKENIZER.encode(t, add_special_tokens=False)
        if len(token_seq) > 512:
            # split the sentence into multiple sentences
            if tok_split:
                token_seqs = [token_seq[i:i+500] for i in range(0, len(token_seq), 500)]
                for ts in token_seqs:
                    split_data.append(TOKENIZER.decode(ts, skip_special_tokens=True))
            else:
                split_data.append(TOKENIZER.decode(token_seq[:500], skip_special_tokens=True))
        else:
            split_data.append(TOKENIZER.decode(token_seq, skip_special_tokens=True))
    return split, split_data

def main():
    # Time taken: 1446.4s -> 24 minutes
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--keys_to_run", nargs='+')
    args = parser.parse_args()
    print(args)
    start = time.time()

    results = []
    for split in args.keys_to_run:
        result = process_split(split)
        results.append(result)

    wmt_year_splits = {split: data for split, data in results}
    print("Time taken: ", time.time() - start)

    print("Writing the data to txt files")
    for split in args.keys_to_run:
        with open(f"./wmt_yearly_data_cut_500_tok/{split}.txt", "w") as f:
            for t in wmt_year_splits[split]:
                f.write(t + "\n")

if __name__ == "__main__":
    main()
