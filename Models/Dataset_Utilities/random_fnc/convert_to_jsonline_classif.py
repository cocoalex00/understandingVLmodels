# Python script to convert a csv to a jasoline 
import argparse
import pandas as pd
import jsonlines

# Main method #
def main():
    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", 
                        help="csv file to be converted to jasonline")
    parser.add_argument("--out", 
                        help="path to output jsonline file")
    
    args = parser.parse_args()
    
    # Load the csv file 
    csvfile = args.csv
    df = pd.read_csv(csvfile, sep=" ")
    
    ### ONLY FOR DEV ###
    df = df.iloc[0:1066,:]

    flist = []
    count = 0
    for index, data in df.iterrows():
        count += 1
        temp = {}
        temp["id"] = count
        temp["img_path"] = str(data[0])
        temp["label"] = int(data[1])

        flist.append(temp)

    with jsonlines.open(args.out, "w") as writer:
        writer.write_all(flist)


if __name__ == "__main__":
    main()