import os
import pathlib
import json
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


def load_queries_query(self):
    with open(self.query_file, encoding='utf8') as fIn:
        for line in fIn:
            line = json.loads(line)
            self.queries[line.get("_id")] = line.get("metadata").get("query")


def load_queries_narrative(self):
    with open(self.query_file, encoding='utf8') as fIn:
        for line in fIn:
            line = json.loads(line)
            self.queries[line.get("_id")] = line.get("metadata").get("narrative")


def load_queries(self):
    with open(self.query_file, encoding='utf8') as fIn:
        for line in fIn:
            line = json.loads(line)
            self.queries[line.get("_id")] = line.get("text")
