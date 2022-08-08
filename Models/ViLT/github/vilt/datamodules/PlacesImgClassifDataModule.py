from vilt.datasets import Places_dataset
from .datamodule_base import BaseDataModule


class PlacesImgClassif_DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return Places_dataset

    @property
    def dataset_name(self):
        return "places"
