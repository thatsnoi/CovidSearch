from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
import random
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def test(dataloader, model_path, sample_size, score_function="dot"):
    corpus, queries, qrels = dataloader.load(split="test")

    if sample_size is not None:
        random.seed(55)
        corpus = dict(random.sample(corpus.items(), sample_size))

    model = DRES(models.SentenceBERT(model_path), batch_size=16)

    retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim"
    results = retriever.retrieve(corpus, queries)

    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
    reranker = Rerank(cross_encoder_model, batch_size=128)

    # Rerank top-100 results retrieved by bi encoder model
    rerank_results = reranker.rerank(corpus, queries, results, top_k=10)

    # Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,5,10,20]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, rerank_results, [1, 5, 10, 20])

    return ndcg, _map, recall, precision
