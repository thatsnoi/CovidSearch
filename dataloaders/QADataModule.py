from typing import Optional
import pandas as pd
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataloaders.CORDDataModule import CORDDataModule
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from generate_queries import generate_queries
from os import path
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.readers import InputExample

def generate_queries_for_dataset(data):
    tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    model.eval()

    df = pd.DataFrame(columns=['query', 'paragraph'])
    for idx in tqdm(range(0, len(data[0:200]))):
        paragraph = data[idx]['abstract'][:512]
        queries = generate_queries(paragraph, model, tokenizer)
        df = pd.concat([df, pd.DataFrame({'query': queries, 'paragraph': [paragraph] * len(queries)})])

    return df


class QADataModule(pl.LightningDataModule):
    """ Data module that generates and holds the question-paragraphs tuples based on the CORD dataset."""
    def __init__(self, data_dir: str = "./../data", dataset: str = "2020-03-13"):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset

    def setup(self, stage: Optional[str] = None) -> None:
        cord_data_module = CORDDataModule(data_dir=self.data_dir)
        cord_data_module.setup()
        train_dataloader = cord_data_module.train_dataloader()
        val_dataloader = cord_data_module.val_dataloader()
        train = train_dataloader.dataset
        val = val_dataloader.dataset

        if path.exists(path.join(self.data_dir, f"{self.dataset}/qa_train.csv")):
            print("Already computed queries for training dataset. Reading file...")
            self.train_qa = pd.read_csv(path.join(self.data_dir, f"{self.dataset}/qa_train.csv"))
        else:
            self.train_qa = generate_queries_for_dataset(train)
            self.train_qa.to_csv(path.join(self.data_dir, f"{self.dataset}/qa_val.csv"), index=False)
        if path.exists(path.join(self.data_dir, f"{self.dataset}/qa_val.csv")):
            print("Already computed queries for validation dataset. Reading file...")
            self.val_qa = pd.read_csv(path.join(self.data_dir, f"{self.dataset}/qa_val.csv"))
        else:
            self.val_qa = generate_queries_for_dataset(val)
            self.val_qa.to_csv(path.join(self.data_dir, f"{self.dataset}/qa_val.csv"), index=False)

    def train_dataloader(self):
        return NoDuplicatesDataLoader([InputExample(texts=[item['query'], item['paragraph']]) for item in self.train_qa.to_dict('records')], batch_size=64)

    def val_dataloader(self):
        return NoDuplicatesDataLoader([InputExample(texts=[item['query'], item['paragraph']]) for item in self.val_qa.to_dict('records')], batch_size=64)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # TODO: Take TREC-COVID test data
        return None


class QADataset(Dataset):
    """ Q&A dataset."""

    def __init__(self, df):
        self.data = {index: InputExample(texts=[item['query'], item['paragraph']]) for index, item in df.to_dict('index').items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    data_module = QADataModule()
    data_module.setup()
    print([InputExample(texts=[item['query'], item['paragraph']]) for item in data_module.train_qa.to_dict('records')])
    #print({index: InputExample(texts=[item, item]) for index, item in data_module.train_qa.to_dict('records')})
    #data_module.train_dataloader()
