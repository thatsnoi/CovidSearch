from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
import random
import logging
from bm25 import bm25
import json
from utils import load_queries_query, load_queries_narrative
import types

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def test(dataloader, model_path, sample_size, score_function="dot"):
    # Using "text" query
    corpus, queries_text, qrels = dataloader.load(split="test")

    # # Using "query" query
    # dataloader.queries = {}
    # dataloader._load_queries = types.MethodType(load_queries_query, dataloader)
    # _, queries_query, _ = dataloader.load(split="test")
    #
    # # Using "narrative" query
    # dataloader.queries = {}
    # dataloader._load_queries = types.MethodType(load_queries_narrative, dataloader)
    # _, queries_narrative, _ = dataloader.load(split="test")

    if sample_size is not None:
        random.seed(55)
        corpus = dict(random.sample(corpus.items(), sample_size))

    model = DRES(models.SentenceBERT(model_path), batch_size=16)

    retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim"

    print('Results with BM25')
    with open('results/results_bm25.json') as json_file:
        results_bm25 = json.load(json_file)
    retriever.evaluate(qrels, results_bm25, [1, 5, 10, 20])

    print('Results for "text" query')
    results_text = retriever.retrieve(corpus, queries_text)
    retriever.evaluate(qrels, results_text, [1, 5, 10, 20])



    # print('Results for "query" query')
    # results_query = retriever.retrieve(corpus, queries_query)
    # retriever.evaluate(qrels, results_query, [1, 5, 10, 20])
    #
    # print('Results for "narrative" query')
    # results_narrative = retriever.retrieve(corpus, queries_narrative)
    # retriever.evaluate(qrels, results_narrative, [1, 5, 10, 20])

    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
    reranker = Rerank(cross_encoder_model, batch_size=128)

    # Fuse rankings
    print("Fusing rankings...")
    fused_rankings = rrf([results_text, results_bm25])
    retriever.evaluate(qrels, fused_rankings, [1, 5, 10, 20])

    # Rerank top-100 results retrieved by bi encoder model
    rerank_results = reranker.rerank(corpus, queries_text, fused_rankings, top_k=20)

    # Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,5,10,20]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, rerank_results, [1, 5, 10, 20])

    return ndcg, _map, recall, precision

import copy


# {'1': {'oq2jgo6z': -19.72185707092285, 'npflq69s': -20.42959213256836, 'lrf75ze3': 27.448469161987305, 'cy4eo4vu': -11.664365768432617}}
def rrf(all_rankings, k=60):
    for rankings_per_topic in all_rankings:
        for topic in rankings_per_topic:
            sorted_rankings = sorted(rankings_per_topic[topic].items(), key=lambda x: -x[1])
            sorted_rankings = [(x[0], idx + 1) for idx, x in enumerate(sorted_rankings)]
            rankings_per_topic[topic] = dict(sorted_rankings)

    fused_rankings = copy.deepcopy(all_rankings)[0]
    for topic in fused_rankings:
        for doc in fused_rankings[topic]:
            fused_rankings[topic][doc] = 0

    for index, rankings_per_topic in enumerate(all_rankings):
        for topic in rankings_per_topic:
            for doc in rankings_per_topic[topic]:
                fused_rankings[topic][doc] = fused_rankings[topic][doc] + (
                            1 / (k + rankings_per_topic[topic][doc]))

    return fused_rankings


if __name__ == '__main__':
    example = [{'1': {'d5': 2.34, 'd4': 2.12, 'd3': 1.93,
                      'd2': 1.43, 'd1': 1.34}}, {
                   '1': {'d5': 1.23, 'd4': 1.02, 'd3': 1.00,
                         'd1': 0.85, 'd2': 0.71}}, {
                   '1': {'d4': 19685, 'd1': 18756, 'd2': 2342,
                         'd5': 2341, 'd3': 123}}]
    rrf(example)
