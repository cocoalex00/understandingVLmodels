# Copyright (c) Alejandro Hernandez Diaz.

# This source code transforms an input csv file to the desired json format:
#   - id: int (id of the data sample)
#   - img_path: str (filename)
#   - label: int (data sample's label)
#   - img_name: str (image's identifier for the tsv file containing the features)


import os
import json
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm 





def _load_dataset(annotations_path,split="train"):  
    """Load entries from a jsonline file

    annotations_path: path to the jsonline file for the dataset
    split: either train, test or val
    """
    df = pd.read_csv(annotations_path, sep=" ", header=None)
    count = 0
    rootpathtrain = "/mnt/fast/nobackup/scratch4weeks/ah02299/trainshmol/data_256"
    rootpathval = "/vol/teaching/HernandezDiazProject/Data/Places365/val_256/"
    rootpathtest = "/vol/teaching/HernandezDiazProject/Data/Places365/test_256"
    newdata = []
    for index, data in tqdm(df.iterrows()):

        if split == "train":
            name = rootpathtrain + str(data[0])  
        elif split == "val":
            name = rootpathval + str(data[0])  
        elif split == "test":
            name = rootpathtest + str(data[0])  


        count += 1
        temp = {}
        temp["id"] = count
        temp["img_path"] = str(data[0])
        #if split != "test":
        temp["label"] = int(data[1])
        temp["img_name"] = name.replace("/","-").split(".")[0]

        newdata.append(temp)
    return split, newdata
    


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--annotations_path",
        type= str,
        help="Path to the csv file to process"
    )
    parser.add_argument(
        "--out_path",
        type= str,
        help="Path to the json file to output"
    )
    parser.add_argument(
        "--split",
        type= str,
        help="split of the dataset being processed"
    )
    args = parser.parse_args()


    split, data = _load_dataset(args.annotations_path, args.split)


    with open(f'{args.out_path}.json', 'w') as g:
        json.dump(data, g, sort_keys=True, indent=4)
    

if __name__ == "__main__":
    main()
