from sklearn.model_selection import train_test_split
import os

# list of all the files in the directory
path = "./wmt_yearly_data_cut_500_tok/"
files = os.listdir(path)
files = [f for f in files if f.endswith(".txt") and "train" in f]

# split the files into train and dev
for f in files:
    file = open(path+f, "r").readlines()
    train, dev = train_test_split(file, test_size=0.05, random_state=123)

    with open(path+"splitted/"+f, "w") as out:
        for line in train:
            out.write(line)

    with open(path+"splitted/"+f.replace("train", "dev"), "w") as out:
        for line in dev:
            out.write(line)