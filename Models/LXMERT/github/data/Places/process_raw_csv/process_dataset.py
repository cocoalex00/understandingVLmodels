import jsonlines
import os
import json
from argparse import ArgumentParser
import pandas as pd


def _load_dataset(annotations_path,split="train"):  
    """Load entries from a jsonline file

    annotations_path: path to the jsonline file for the dataset
    split: either train, test or val
    """
    df = pd.read_csv(annotations_path, sep=" ")
    count = 0

    newdata = []
    for index, data in df.iterrows():
        count += 1
        temp = {}
        temp["id"] = count
        temp["img_path"] = str(data[0])
        temp["label"] = int(data[1])
        temp["img_name"] = str(data[0].replace("/","-").split(".")[0])

        newdata.append(temp)
    return split, newdata
    


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--anotations_path",
        type= str,
        help="Path to the csv file to process"
    )

    args = parser.parse_args()
    split, data = _load_dataset(args.anotations_path)

    print(data)

    with open('../%s.json' % split, 'w') as g:
        json.dump(data, g, sort_keys=True, indent=4)

if __name__ == "__main__":
    main()
