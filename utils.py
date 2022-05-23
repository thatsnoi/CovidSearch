import os
import pathlib

from beir import util
from beir.datasets.data_loader import GenericDataLoader


def download_dataset():
    dataset = "trec-covid"

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    # Provide the data_path where the corpus has been downloaded and unzipped
    dataloader = GenericDataLoader(data_folder=data_path)

    return dataloader, data_path
