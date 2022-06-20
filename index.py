from beir.retrieval import models
from beir.retrieval.search.dense import FlatIPFaissSearch
import os


def index(dataloader, model_path):
    corpus, queries_text, qrels = dataloader.load(split="test")
    model = models.SentenceBERT(model_path)
    faiss_search = FlatIPFaissSearch(model)

    faiss_search.index(corpus)

    output_dir = "./faiss-index"
    os.makedirs(output_dir, exist_ok=True)
    faiss_search.save(output_dir=output_dir, prefix="my-index", ext="default")
