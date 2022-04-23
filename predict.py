import pandas as pd
import numpy as np
import time
import faiss

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')

df = pd.read_csv("./data/2020-03-13/preprocessed.csv", index_col=0)

index = faiss.read_index('./data/2020-03-13/cord.index')


def fetch_paper_info(dataframe_idx):
    info = df.iloc[dataframe_idx]
    meta_dict = dict()
    meta_dict['title'] = info['title']
    meta_dict['abstract'] = info['abstract'][:500]
    return meta_dict


def search(query, top_k, index, model):
    print("Query: " + query)
    t = time.time()
    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k)
    print('>>>> Results in Total Time: {}'.format(time.time() - t))
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results = [fetch_paper_info(idx) for idx in top_k_ids]
    return results


if __name__ == "__main__":
    query = "What is the most common covid-19 symptom?"
    results = search(query, top_k=5, index=index, model=model)
    print("\n")
    for result in results:
        print('\t', result)
