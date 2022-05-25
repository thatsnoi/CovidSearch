from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
import random
import logging
from utils import load_queries_query, load_queries_narrative
import types

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)



def test(dataloader, model_path, sample_size, score_function="dot"):
    # Using "text" query
    corpus, queries_text, qrels = dataloader.load(split="test")

    # Using "query" query
    dataloader.queries = {}
    dataloader._load_queries = types.MethodType(load_queries_query, dataloader)
    _, queries_query, _ = dataloader.load(split="test")

    # Using "narrative" query
    dataloader.queries = {}
    dataloader._load_queries = types.MethodType(load_queries_narrative, dataloader)
    _, queries_narrative, _ = dataloader.load(split="test")


    if sample_size is not None:
        random.seed(55)
        corpus = dict(random.sample(corpus.items(), sample_size))

    model = DRES(models.SentenceBERT(model_path), batch_size=16)

    retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim"

    print('Results for "text" query')
    results_text = retriever.retrieve(corpus, queries_text)
    retriever.evaluate(qrels, results_text, [1, 5, 10, 20])

    print('Results for "query" query')
    results_query = retriever.retrieve(corpus, queries_query)
    retriever.evaluate(qrels, results_query, [1, 5, 10, 20])

    print('Results for "narrative" query')
    results_narrative = retriever.retrieve(corpus, queries_narrative)
    retriever.evaluate(qrels, results_narrative, [1, 5, 10, 20])

    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
    reranker = Rerank(cross_encoder_model, batch_size=128)

    # Rerank top-100 results retrieved by bi encoder model
    rerank_results = reranker.rerank(corpus, queries_text, results_text, top_k=100)

    # Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,5,10,20]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, rerank_results, [1, 5, 10, 20])

    return ndcg, _map, recall, precision
