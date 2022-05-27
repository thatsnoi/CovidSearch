from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
import random
import logging
import json

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def test(dataloader, model_path, sample_size, score_function="dot", cross_encoder=True,
         fuse_with_bm25=True, cross_encoder_top_k=20):
    # Using "text" query
    corpus, queries_text, qrels = dataloader.load(split="test")

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
    results = retriever.retrieve(corpus, queries_text)
    retriever.evaluate(qrels, results, [1, 5, 10, 20])

    # Fuse rankings
    if fuse_with_bm25:
        print("Fusing rankings...")
        results = rrf([results, results_bm25])
        retriever.evaluate(qrels, results, [1, 5, 10, 20])

    if cross_encoder:
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
        reranker = Rerank(cross_encoder_model, batch_size=128)

        # Rerank top-100 results retrieved by bi encoder model
        results = reranker.rerank(corpus, queries_text, results, top_k=cross_encoder_top_k)

    # Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,5,10,20]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 5, 10, 20])

    return ndcg, _map, recall, precision


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

