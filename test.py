import random

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

import logging
import pathlib, os

#### Just some code to print debug information to stdout
from sentence_transformers import SentenceTransformer

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "trec-covid"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#random.seed(55)
#corpus = dict(random.sample(corpus.items(), 1000))

#### Load the SBERT model and retrieve using cosine-similarity
#model = DRES(models.SentenceBERT("msmarco-distilbert-base-v3"), batch_size=16)

model = DRES(models.SentenceBERT('./output/bert-base-uncased-GenQ-nfcorpus'), batch_size=16)

retriever = EvaluateRetrieval(model, score_function="dot") # or "dot" for dot-product
results = retriever.retrieve(corpus, queries)


cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
reranker = Rerank(cross_encoder_model, batch_size=128)

# Rerank top-100 results retrieved by bi encoder model
rerank_results = reranker.rerank(corpus, queries, results, top_k=10)


#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
#ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 3, 5, 10])

ndcg, _map, recall, precision = retriever.evaluate(qrels, rerank_results, [1, 3, 5, 10])

if __name__ == "__main__":
   print("Do some testing here if needed.")


# 2022-05-14 13:23:00 -
#
# 2022-05-14 13:23:00 - NDCG@1: 0.2800
# 2022-05-14 13:23:00 - NDCG@3: 0.2059
# 2022-05-14 13:23:00 - NDCG@5: 0.1602
# 2022-05-14 13:23:00 - NDCG@10: 0.1293
# 2022-05-14 13:23:00 -
#
# 2022-05-14 13:23:00 - MAP@1: 0.0008
# 2022-05-14 13:23:00 - MAP@3: 0.0013
# 2022-05-14 13:23:00 - MAP@5: 0.0015
# 2022-05-14 13:23:00 - MAP@10: 0.0017
# 2022-05-14 13:23:00 -
#
# 2022-05-14 13:23:00 - Recall@1: 0.0008
# 2022-05-14 13:23:00 - Recall@3: 0.0015
# 2022-05-14 13:23:00 - Recall@5: 0.0018
# 2022-05-14 13:23:00 - Recall@10: 0.0027
# 2022-05-14 13:23:00 -
#
# 2022-05-14 13:23:00 - P@1: 0.3400
# 2022-05-14 13:23:00 - P@3: 0.2133
# 2022-05-14 13:23:00 - P@5: 0.1520
# 2022-05-14 13:23:00 - P@10: 0.1200
# Do some testing here if needed.



### OUR MODEL
# 2022-05-14 13:31:23 - NDCG@1: 0.2400
# 2022-05-14 13:31:23 - NDCG@3: 0.1729
# 2022-05-14 13:31:23 - NDCG@5: 0.1432
# 2022-05-14 13:31:23 - NDCG@10: 0.1104
# 2022-05-14 13:31:23 -
#
# 2022-05-14 13:31:23 - MAP@1: 0.0006
# 2022-05-14 13:31:23 - MAP@3: 0.0010
# 2022-05-14 13:31:23 - MAP@5: 0.0012
# 2022-05-14 13:31:23 - MAP@10: 0.0014
# 2022-05-14 13:31:23 -
#
# 2022-05-14 13:31:23 - Recall@1: 0.0006
# 2022-05-14 13:31:23 - Recall@3: 0.0012
# 2022-05-14 13:31:23 - Recall@5: 0.0014
# 2022-05-14 13:31:23 - Recall@10: 0.0022
# 2022-05-14 13:31:23 -
#
# 2022-05-14 13:31:23 - P@1: 0.2600
# 2022-05-14 13:31:23 - P@3: 0.1800
# 2022-05-14 13:31:23 - P@5: 0.1400
# 2022-05-14 13:31:23 - P@10: 0.1000


## dot
# 2022-05-14 13:47:57 - NDCG@1: 0.2700
# 2022-05-14 13:47:57 - NDCG@3: 0.1863
# 2022-05-14 13:47:57 - NDCG@5: 0.1578
# 2022-05-14 13:47:57 - NDCG@10: 0.1163


## rerank

# 2022-05-14 14:20:04 - NDCG@1: 0.3700
# 2022-05-14 14:20:04 - NDCG@3: 0.2539
# 2022-05-14 14:20:04 - NDCG@5: 0.1972
# 2022-05-14 14:20:04 - NDCG@10: 0.1339