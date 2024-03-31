from datasets import load_dataset
from transformers import BertTokenizer
from multiprocessing import Pool, cpu_count

def process_split(split):
    tokenizer = BertTokenizer.from_pretrained("facebook/contriever")
    wmt_year_data = load_dataset("KaiNylund/WMT-year-splits", split=split)

    split_data = []
    for t in wmt_year_data['text']:
        token_seq = tokenizer.encode(t, add_special_tokens=False)
        if len(token_seq) > 512:
            # split the sentence into multiple sentences
            token_seqs = [token_seq[i:i+500] for i in range(0, len(token_seq), 500)]
            for ts in token_seqs:
                split_data.append(tokenizer.decode(ts, skip_special_tokens=True))
        else:
            split_data.append(tokenizer.decode(token_seq, skip_special_tokens=True))
    return split, split_data

def main():
    wmt_year_data = load_dataset("KaiNylund/WMT-year-splits")

    # Initialize multiprocessing Pool
    pool = Pool(cpu_count())

    # Use pool.map to process data in parallel
    results = pool.map(process_split, wmt_year_data.keys())

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    wmt_year_splits = {split: data for split, data in results}

    print("Writing the data to txt files")
    for split in wmt_year_splits:
        with open(f"./wmt_yearly_data/{split}.txt", "w") as f:
            for t in wmt_year_splits[split]:
                f.write(t + "\n")

if __name__ == "__main__":
    main()
