from .base_dataset import BaseDataset
import sys
import random


class Places365(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["train"]
        elif split == "val":
            names = ["val"]
        elif split == "test":
            names = ["test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name=None,
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        result = None
        while result is None:
            try:
                image = self.get_image(index)["image"]

                result = True
            except:
                print(
                    f"error while read file idx {index} in {self.names[0]}",
                    file=sys.stderr,
                )
                index = random.randint(0, len(self.index_mapper) - 1)
            label = self.table["label"][index]

        return {
            "image": image,
            "table_name": self.table_names[index],
            "label": label
        }
