import pandas as pd
from fastapi import FastAPI
from utils import download_dataset, load_faiss, index_bm25, get_results_bm25, rrf
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


faiss_search = load_faiss()
print("Loaded faiss")
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
print("Loaded ce model")
reranker = Rerank(cross_encoder_model, batch_size=1)
dataloader, data_path = download_dataset()
df = pd.read_csv('./datasets/cord/metadata.csv')
df = df.fillna('')

print("Downloaded dataset")
index_bm25(dataloader)
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
    ce: bool
    top_k: int
    fuse: bool

def get_metadata(doc_id):
    doc = df.loc[df['cord_uid'] == doc_id]
    return doc[['cord_uid', 'title', 'doi', 'source_x', 'abstract', 'publish_time', 'authors', 'url']].iloc[0].to_dict()

@app.post("/search")
def search(data: Data):
    query = data.query

    cross_encoder_k = data.top_k

    print("Retrieving similar docs")
    results = faiss_search.search(corpus, {'1': query}, 100, "dot")

    if data.fuse:
        results_bm25 = get_results_bm25(query)
        results = rrf([results_bm25, results])

    if data.ce and cross_encoder_k > 0:
        # Rerank top-k results retrieved by bi encoder model
        results_before = sorted(results['1'].items(), key=lambda item: item[1], reverse=True)
        print("Applying cross-encoder")
        results = reranker.rerank(corpus, {'1': query}, results, top_k=cross_encoder_k)
        results_after = sorted(results['1'].items(), key=lambda item: item[1], reverse=True)
        scores_sorted = results_after[:cross_encoder_k] + results_before[cross_encoder_k:]
    else:
        scores_sorted = sorted(results['1'].items(), key=lambda item: item[1], reverse=True)

    results_with_metadata = []
    for rank in range(len(scores_sorted)):
        doc_id = scores_sorted[rank][0]
        # Format: Rank x: ID [Title] Body
        # print("Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
        # print(corpus[doc_id])
        results_with_metadata.append(get_metadata(doc_id))

    return results_with_metadata

