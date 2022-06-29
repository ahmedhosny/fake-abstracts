import pandas as pd
import csv
from datasets import Dataset, DatasetDict
import numpy as np 
import os

# FILE = "pubmed-deeplearni-set.txt"
# f = open(FILE, "r")

PATH = "/data/pubmed_data"
PMID_TAG = "PMID- "
TITLE_TAG = "TI  - "
ABSTRACT_TAG = "AB  - "
SPACE = "      "
papers = []

filenames = os.listdir(PATH)

contents = ""
for filename in filenames:
    with open(os.path.join(PATH, filename)) as f:
        contents += f.read()

print (len(contents.split("\n\n")))

def get_entire_string(idx, lines, tag):
    entire_string = lines[idx].replace(tag, "")
    for line in lines[idx+1:]:
            if line.startswith(SPACE):
                entire_string += line.replace(SPACE, "")
            else:
                break
    return entire_string

# broken now
def get_dataset():

    chuncks = f.read().split("\n\n")
    for chunk in chuncks:
        lines = chunk.split('\n')
        title = ""
        abstract = ""
        pmid = ""
        for idx, line in enumerate(lines):
            if line.startswith(TITLE_TAG):
                title = get_entire_string(idx, lines, TITLE_TAG)
            elif line.startswith(ABSTRACT_TAG):
                abstract = get_entire_string(idx, lines, ABSTRACT_TAG)
            elif line.startswith(PMID_TAG):
                pmid = get_entire_string(idx, lines, PMID_TAG)
        
        _obj = {"title": title, "abstract": abstract, "pmid":int(pmid)}
        if title != "" and abstract != "" and pmid != "":
            papers.append(_obj)
        
        # if idx == 80:
        #     print(_obj)


    df = pd.DataFrame(papers)
    title_lengths = [len(x) for x in df["title"]]
    abstract_lengths = [len(x) for x in df["abstract"]]

    print (f"95% title chacacters :: {np.percentile(title_lengths, 95)}")
    print (f"95% abstract chacacters :: {np.percentile(abstract_lengths, 95)}")


    ds = DatasetDict({
        "train": Dataset.from_pandas(df[:8000]),
        "validation": Dataset.from_pandas(df[8000:9000]),
        "test": Dataset.from_pandas(df[9000:])
    })

    return ds


def get_dataset_for_gpt2():
    out_dict = {}

    chuncks = contents.split("\n\n")
    for chunk in chuncks:
        lines = chunk.split('\n')
        title = ""
        abstract = ""
        pmid = ""
        for idx, line in enumerate(lines):
            if line.startswith(TITLE_TAG):
                title = get_entire_string(idx, lines, TITLE_TAG)
            elif line.startswith(ABSTRACT_TAG):
                abstract = get_entire_string(idx, lines, ABSTRACT_TAG)
            elif line.startswith(PMID_TAG):
                pmid = get_entire_string(idx, lines, PMID_TAG)
        
        if title != "" and abstract != "" and pmid != "":
            out_dict[int(pmid)] = [title, abstract]

    return out_dict
        
  
data = get_dataset_for_gpt2()
print (len(data.keys()))
print (len(data.values()))
for i in [0, 50000, -1]:
    print (i)
    print (list(data.keys())[i])
    print (list(data.values())[i])



