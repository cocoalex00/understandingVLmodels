from .sapeplaces import PlacesDatasetBase
import sys
import random
import torch


class Places365(PlacesDatasetBase):
    def __init__(self, datadir, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["places_train"]
        elif split == "val":
            names = ["places_val"]
        elif split == "test":
            names = ["places_test"]

        super().__init__(
            data_dir=datadir,
            names=names,
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        result = None
        while result is None:
            try:
                image = self.get_image(index)["image"]
                text = self.get_text(index)["text"]
                result = True
            except:
                print(
                    f"error while read file idx {index} in {self.names[0]}",
                    file=sys.stderr,
                )
                index = random.randint(0, len(self.index_mapper) - 1)
            label = self.table["label"][index].as_py()
            text = self.get_text(index)["text"]

        return {
            "image": image,
            "table_name": self.table_names[index],
            "label": label,
            "text": text
        }
