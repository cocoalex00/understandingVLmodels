import json
import pandas as pd
import pyarrow as pa
import os

from tqdm import tqdm
from collections import defaultdict


def process(root, iden, row):
    #texts = ["[CLS]]"for r in row]
    labels = [r["label"] for r in row]

    split = iden.split("-")[0]
    print(row)
    path = row[0]["img_path"]
    identifier = row[0]["img_path"].replace("/","-").split(".")[0] 
    text = " "

    with open(os.path.join(root,path), "rb") as fp:
        img= fp.read()


    return [img, labels, identifier,text]


def make_arrow(root, dataset_root,split):

    if split == "train":
        train_data = list(
            map(json.loads, open(f"{root}/places365_train_alexsplit.json").readlines())
        )
        val_data = None
        test_data = None

    elif split =="val":
        val_data = list(
            map(json.loads, open(f"{root}/places365_val.json").readlines())
        )

        test_data = None
        train_data = None
    elif split == "test":
        test_data = list(map(json.loads, open(f"{root}/places365_test.json").readlines()))

        train_data = None
        val_data = None


    datas = {
        "train":  train_data,
        "val": val_data,
        "test": test_data,
    }
      


    annotations = dict()

    data = datas[split]

    _annot = defaultdict(list)
    for row in tqdm(data):

        _annot[row["img_path"].replace("/","-").split(".")[0]].append(row)
    annotations[split] = _annot
    bs = [
        process(root, split, row) for iden, row in tqdm(annotations[split].items())
    ]

    dataframe = pd.DataFrame(
        bs, columns=["img", "label", "identifier", "text"],
    )

    table = pa.Table.from_pandas(dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(f"{dataset_root}/places_{split}.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


if __name__ == "__main__":

    make_arrow("/mnt/c/Users/aleja/Desktop/MSc Project/images/totest/","/mnt/c/Users/aleja/Desktop/MSc Project/images","val")
   