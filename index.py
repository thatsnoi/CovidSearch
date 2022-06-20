from beir.retrieval import models
from beir.retrieval.search.dense import FlatIPFaissSearch
import os

from utils import download_dataset
import neptune.new as neptune
from zipfile import ZipFile


def index(neptune_id="TREC-66", model_path="./output/biobert-GenQ-bi-encoder-GenQ-bi-encoder"):
    dataloader, data_path = download_dataset()

    print(f"Downloading bi-encoder from neptune project {neptune_id}")
    project = neptune.init(
        project="noahjadallah/TREC-Covid",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzgwNzhlNy1iMDVlLTQwNWUtYWJlYS04NWMxNjA0YmQ3ODAifQ==",
        run=neptune_id,
        mode="read-only"
    )
    project["model/bi-encoder"].download()
    project.stop()

    with ZipFile('bi-encoder.zip', 'r') as zipObj:
        zipObj.extractall()

    corpus, queries_text, qrels = dataloader.load(split="test")
    model = models.SentenceBERT(model_path)
    faiss_search = FlatIPFaissSearch(model)

    faiss_search.index(corpus)

    output_dir = "./faiss-index"
    os.makedirs(output_dir, exist_ok=True)
    faiss_search.save(output_dir=output_dir, prefix="my-index", ext="default")
