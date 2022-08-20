from .base_retrieval import PlacesDatasetBaseRetrieval
import sys
import random
import torch
import traceback


class Places365(PlacesDatasetBaseRetrieval):
    def __init__(self, datadir,labels_path, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["places_val"]
            val = False 
        elif split == "val":
            names = ["placesretrieval_val"]
            val = True
        elif split == "test":
            names = ["places_test"]
            val = False

        super().__init__(
            data_dir=datadir,
            labels_path = labels_path,
            names=names,
            remove_duplicate=False,
            val = val
        )

    def __getitem__(self, index):
        result = None
        while result is None:
            try:        
                image = self.get_image(index)["image"]
                textsample = self.get_text(index)
                label = textsample["label"]
                text = textsample["text"]
                result = True
            except: 
                traceback.print_exc()
                sys.exit(1)
                print(
                    f"error while read file idx {index} in {self.names[0]}",
                    file=sys.stderr,
                )
                index = random.randint(0, len(self.index_mapper) - 1)


        return {
            "image": image,
            "table_name": self.table_names[index],
            "label": label,
            "text": text
        }
