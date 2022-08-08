# Python script to convert a list of 
import argparse
import pandas as pd
import jsonlines
import csv

def _translate_object_ids(object_tags, object_ids):

    translated_tags = []
    for id in object_ids:
        translated_tags.append(object_tags[int(id) ])
    
    return translated_tags

# Main method #
def main():
    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", 
                        default="./object_tags/objects_vocab.txt",
                        help="csv file containing the list of object tags")
    parser.add_argument("--ids", 
                        nargs="+",
                        default= [159,  453,  314,  200,   70,  159,  314,  200,  314,  758,  128,  248,
         177,  200,  106, 1409, 1260,  314,  829,  314, 1523,  314,  159,  248,
          70,  314,  808,  453,  159,  200, 1409,  808, 1409,  177,   70,  128],
                        help="list of object ids to translate")
    
    args = parser.parse_args()
    
    # get list of object tags
    object_tags = []
    with open(args.csv, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            object_tags.append(row)
    
    translated_tags = _translate_object_ids(object_tags, set(args.ids))

    print(f"object ids to be translated: {set(args.ids)}")
    print(f"translated object tags: {translated_tags}")
    


if __name__ == "__main__":
    main()