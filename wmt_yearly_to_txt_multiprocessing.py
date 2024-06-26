import os 
os.system("taskset -p 0xfffff %d" % os.getpid())

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
    for t in wmt_year_data['text']:
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
    import time
    start = time.time()

    # Initialize multiprocessing Pool
    # pool = Pool(cpu_count())

    print(cpu_count())

    # Use pool.map to process data in parallel
    with Pool(processes=5) as pool:
        # results = tqdm(pool.map(process_split, wmt_year_data.keys()), total=len(wmt_year_data.keys()))
        results = list(tqdm(pool.imap(process_split, WMT_YEAR_DATA.keys()), total=len(WMT_YEAR_DATA.keys())))

    # Close the pool and wait for the work to finish
    # pool.close()
    # pool.join()

    wmt_year_splits = {split: data for split, data in results}
    print("Time taken: ", time.time() - start)

    print("Writing the data to txt files")
    for split in wmt_year_splits:
        with open(f"./wmt_yearly_data_cut_500_tok/{split}.txt", "w") as f:
            for t in wmt_year_splits[split]:
                f.write(t + "\n")

if __name__ == "__main__":
    main()
