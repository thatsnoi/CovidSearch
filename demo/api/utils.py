import os
import pathlib
from zipfile import ZipFile
#
import neptune.new as neptune
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import FlatIPFaissSearch
import random
import json
import requests


def download_dataset():
    dataset = "trec-covid"

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    # Provide the data_path where the corpus has been downloaded and unzipped
    dataloader = GenericDataLoader(data_folder=data_path)

    return dataloader, data_path


def load_faiss():
    model = SentenceBERT("./bi-encoder")
    faiss_search = FlatIPFaissSearch(model)

    prefix = "my-index"
    ext = "default"
    input_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")

    if os.path.exists(os.path.join(input_dir, "{}.{}.faiss".format(prefix, ext))):
        faiss_search.load(input_dir=input_dir, prefix=prefix, ext=ext)

    return faiss_search


def index_bm25(dataloader, sample_size=None, data_path="./datasets/trec-covid",
               docker_beir_pyserini="http://bm25:8000"):
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

    # Upload Multipart-encoded files
    with open(os.path.join(data_path, "pyserini.jsonl"), "rb") as fIn:
        r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

    # Index documents to Pyserini #####
    index_name = "beir/trec-covid"  # beir/scifact
    r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})
    print("Finished indexing")


def get_results_bm25(query, docker_beir_pyserini="http://bm25:8000"):
    # Retrieve documents from Pyserini #####
    payload = {"queries": [query], "qids": ["1"], "k": 100}

    # Retrieve pyserini results (format of results is identical to qrels)
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

    return results


def rrf(all_rankings, k=60):
    for rankings_per_topic in all_rankings:
        for topic in rankings_per_topic:
            sorted_rankings = sorted(rankings_per_topic[topic].items(), key=lambda x: -x[1])
            sorted_rankings = [(x[0], idx + 1) for idx, x in enumerate(sorted_rankings)]
            rankings_per_topic[topic] = dict(sorted_rankings)

    fused_rankings = {}

    for index, rankings_per_topic in enumerate(all_rankings):
        for topic in rankings_per_topic:
            if topic not in fused_rankings:
                fused_rankings[topic] = {}
            for doc in rankings_per_topic[topic]:
                if doc not in fused_rankings[topic]:
                    fused_rankings[topic][doc] = 0
                fused_rankings[topic][doc] = fused_rankings[topic][doc] + (
                        1 / (k + rankings_per_topic[topic][doc]))

    return fused_rankings
