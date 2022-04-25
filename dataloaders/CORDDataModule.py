from typing import Optional
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import tarfile
from os import path
import wget
import nltk
import pandas as pd
import ssl
import numpy as np
# import glob
# import json
import faiss
from sentence_transformers import SentenceTransformer

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')


class CORDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./../data", dataset: str = "2020-03-13", ask_for_recompute=False):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.ask_for_recompute = ask_for_recompute

    def prepare_data(self) -> None:
        self.download_data()
        self.extract_data()
        self.preprocess_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.data = self.preprocess_data()
        train, test = train_test_split(self.data, test_size=0.1)
        train, val = train_test_split(self.data, test_size=0.2)
        self.train = train
        self.val = val
        self.test = test

    def download_data(self):
        if path.exists(path.join(self.data_dir, f"cord-19_{self.dataset}.tar.gz")):
            print("Data already on disk, skipping downloading.")
        else:
            print("Downloading dataset...")
            wget.download(
                f"https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_{self.dataset}.tar.gz",
                self.data_dir
            )
            print("Successfully downloaded data.")

    def extract_data(self):
        if path.exists(path.join(self.data_dir, self.dataset)):
            print("Data already extracted, skipping extraction.")
        else:
            print("Extracting data...")
            file = tarfile.open(path.join(self.data_dir, f"cord-19_{self.dataset}.tar.gz"))
            file.extractall(self.data_dir)
            file.close()

            file = tarfile.open(path.join(self.data_dir, f"{self.dataset}/comm_use_subset.tar.gz"))
            file.extractall(path.join(self.data_dir, self.dataset))
            file.close()
            print("Successfully extracted data.")

    def preprocess_data(self):
        if path.exists(path.join(self.data_dir, self.dataset + "/preprocessed.csv")):
            print("Data already preprocessed, reading file.")
            data = pd.read_csv(path.join(self.data_dir, self.dataset + "/preprocessed.csv"), index_col=0)
        else:
            print("Preprocessing data...")
            data = pd.read_csv(path.join(self.data_dir, f'{self.dataset}/all_sources_metadata_{self.dataset}.csv'))

            # print("Joining datasets...")
            # folder_path = path.join(self.data_dir, self.dataset + "/comm_use_subset")
            # text_df = pd.DataFrame(columns=['paper_id', 'body'])
            # for filename in glob.glob(path.join(folder_path, '*.json')):
            #     with open(filename, 'r') as f:
            #         text = f.read()
            #         doc = json.loads(text)
            #         doc_processed = {
            #             'paper_id': [doc['paper_id']],
            #             'body': ["\n".join([paragraph['text'] for paragraph in doc['body_text']])]
            #         }
            #
            #         text_df = pd.concat([text_df, pd.DataFrame(doc_processed)])
            #
            # data = data.merge(text_df, left_on='sha', right_on='paper_id')

            print("Tokenizing data...")
            data = data.dropna(subset=["abstract", "sha"])
            data = data.drop_duplicates(subset=["sha"])
            data['abstract'] = data['abstract'].apply(lambda x: x.replace('<jats:p>', ''))
            data['abstract_tokens'] = [nltk.word_tokenize(text) for text in data['abstract']]
            data = data[['sha', 'title', 'abstract', 'abstract_tokens']]
            # data['abstract_tokens'] = [[nltk.word_tokenize(sentence) for sentence in nltk.sent_tokenize(text)] for text in data['abstract']]
            # data['title_tokens'] = [nltk.word_tokenize(text) for text in data['title']]
            # data['body_tokens'] = [[nltk.word_tokenize(sentence) for sentence in nltk.sent_tokenize(text)] for text in
            #                        data['body']]
            # data['body_tokens_flattened'] = [nltk.word_tokenize(text) for text in data['body']]

            # Save preprocessed file
            data.to_csv(path.join(self.data_dir, f"{self.dataset}/preprocessed.csv"))

        if path.exists(path.join(self.data_dir, self.dataset + "/cord.index")):
            if self.ask_for_recompute:
                print("Already computed embeddings. Type 'y' if you want to recompute: ")
                recompute = input()
                if recompute != 'y':
                    return data
            else:
                print("Already computed embeddings.")
                return data

        print("Computing embeddings...")
        model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')

        encoded_data = model.encode(data.abstract_tokens.tolist()[0:1000])
        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        index.add_with_ids(encoded_data, np.array(range(0, len(encoded_data))))
        faiss.write_index(index, 'cord.index')

        return data

    def train_dataloader(self):
        return DataLoader(CORDDataset(self.train))

    def val_dataloader(self):
        return DataLoader(CORDDataset(self.val))

    def test_dataloader(self):
        return DataLoader(CORDDataset(self.test))


class CORDDataset(Dataset):
    """ CORD dataset."""

    def __init__(self, df):
        self.data = df.to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    data_module = CORDDataModule()
    data_module.prepare_data()
