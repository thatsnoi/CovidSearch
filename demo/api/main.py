from typing import Union
from fastapi import FastAPI, Form
from utils import download_dataset, load_faiss, index_bm25, get_results_bm25
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

faiss_search = load_faiss()
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
reranker = Rerank(cross_encoder_model, batch_size=1)
dataloader, data_path = download_dataset()
# index_bm25(dataloader)
corpus, _, _ = dataloader.load(split="test")
# corpus = {k: v for k, v in corpus.items() if len(v.get("text")) > 10}

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Data(BaseModel):
    query: str


@app.post("/search")
def search(data: Data):
    query = data.query

    cross_encoder_k = 20

    print("Retrieving similar docs")
    results = faiss_search.search(corpus, {'1': query}, 100, "dot")

    print("Applying cross-encoder")
    if cross_encoder_k > 0:
        # Rerank top-100 results retrieved by bi encoder model
        results = reranker.rerank(corpus, {'1': query}, results, top_k=cross_encoder_k)

    scores_sorted = sorted(results['1'].items(), key=lambda item: item[1], reverse=True)

    results_with_metadata = []
    for rank in range(cross_encoder_k):
        doc_id = scores_sorted[rank][0]
        # Format: Rank x: ID [Title] Body
        print("Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
        # print(corpus[doc_id])
        results_with_metadata.append({
            'doc_id': doc_id,
            'title': corpus[doc_id].get("title"),
            'text': corpus[doc_id].get("text")
        })

    return results_with_metadata


if __name__ == '__main__':
    cross_encoder_k = 10
    query = 'how does the coronavirus respond to changes in the weather'

    print("Retrieving similar docs")
    results = faiss_search.search(corpus, {'1': query}, 100, "dot")

    print("Applying cross-encoder")
    if cross_encoder_k > 0:
        # Rerank top-100 results retrieved by bi encoder model
        results = reranker.rerank(corpus, {'1': query}, results, top_k=cross_encoder_k)

    scores_sorted = sorted(results['1'].items(), key=lambda item: item[1], reverse=True)

    results_with_metadata = []
    for rank in range(cross_encoder_k):
        doc_id = scores_sorted[rank][0]
        # Format: Rank x: ID [Title] Body
        print("Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
        # print(corpus[doc_id])
        results_with_metadata.append({
            'doc_id': doc_id,
            'title': corpus[doc_id].get("title"),
            'text': corpus[doc_id].get("text")
        })
    print(results_with_metadata)
