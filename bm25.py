"""
After docker installation, please follow the steps below to get docker container up and running:
1. docker pull beir/pyserini-fastapi 
2. docker run -p 8000:8000 -it --rm beir/pyserini-fastapi
Once the docker container is up and running in local, now run the code below.
This code doesn't require GPU to run.
Usage: python evaluate_anserini_bm25.py
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from utils import download_dataset

import pathlib, os, json
import logging
import requests
import random
from os import path

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def bm25(dataloader, sample_size=None, data_path="./datasets/trec-covid"):
    corpus, queries, qrels = dataloader.load(split="test")

    if sample_size is not None:
        random.seed(55)
        corpus = dict(random.sample(corpus.items(), sample_size))

    # Convert BEIR corpus to Pyserini Format #####
    pyserini_jsonl = "pyserini.jsonl"
    with open(os.path.join(data_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in corpus:
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            data = {"id": doc_id, "title": title, "contents": text}
            json.dump(data, fOut)
            fOut.write('\n')

    # Download Docker Image beir/pyserini-fastapi
    # Locally run the docker Image + FastAPI ####
    docker_beir_pyserini = "http://127.0.0.1:8000"

    # Upload Multipart-encoded files
    with open(os.path.join(data_path, "pyserini.jsonl"), "rb") as fIn:
        r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

    # Index documents to Pyserini #####
    index_name = "beir/trec-covid"  # beir/scifact
    r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})
    print("Finished indexing")

    # Retrieve documents from Pyserini #####
    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}

    # Retrieve pyserini results (format of results is identical to qrels)
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

    logging.info("Retriever evaluation for k in: {}".format([1, 3, 5, 10]))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 3, 5, 10])

    with open(path.join(data_path, 'results_bm25.json'), 'w') as outfile:
        json.dump(results, outfile)

    return results

if __name__ == '__main__':
    dataloader, data_path = download_dataset()
    bm25(dataloader)