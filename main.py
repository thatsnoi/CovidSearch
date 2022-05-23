import argparse
from utils import download_dataset
from query_gen_and_train import gen_queries, train_bi_encoder
from test import test

parser = argparse.ArgumentParser(description='Parameters for TREC-Covid IR.')
parser.add_argument('--sample_size',        type=int, default=None, help='Corpus sample size')
parser.add_argument('--gen',                type=bool, default=True, help='Generate queries')
parser.add_argument('--pretrained_model',   type=str, default=True,
                    help='Pretrained huggingface model for bi-encoder training.')
parser.add_argument('--num_epochs',             type=int, default=10, help='Epochs for bi-encoder training.')


args = parser.parse_args()

dataloader, data_path = download_dataset()

gen_queries(dataloader, data_path, sample_size=args.sample_size)
model_save_path = train_bi_encoder(data_path, num_epochs=args.num_epochs)

test(dataloader, model_save_path, args.sample_size)