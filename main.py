import argparse
from utils import download_dataset
from query_gen_and_train import gen_queries, train_bi_encoder
from test import test
import neptune.new as neptune

parser = argparse.ArgumentParser(description='Parameters for TREC-Covid IR.')
parser.add_argument('--name',               type=str, help='Experiment name for logging')
parser.add_argument('--sample_size',        type=int, default=None, help='Corpus sample size')
parser.add_argument('--gen',                type=bool, default=True, help='Generate queries')
parser.add_argument('--pretrained_model',   type=str, default='dmis-lab/biobert-v1.1',
                    help='Pretrained huggingface model for bi-encoder training.')
parser.add_argument('--num_epochs',         type=int, default=10, help='Epochs for bi-encoder training.')


args = parser.parse_args()

dataloader, data_path = download_dataset()

run = neptune.init(
    project="noahjadallah/TREC-Covid",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzgwNzhlNy1iMDVlLTQwNWUtYWJlYS04NWMxNjA0YmQ3ODAifQ==",
)

params = {
    "pretrained_model": args.pretrained_model,
    "sample_size": args.sample_size,
    "num_epochs": args.num_epochs
    }
run["parameters"] = params
run["sys/name"] = args.name

if args.gen:
    gen_queries(dataloader, data_path, sample_size=args.sample_size)
else:
    print("Skipped generating queries")

model_save_path = train_bi_encoder(data_path, num_epochs=args.num_epochs, model_name=args.pretrained_model)

ndcg, _map, _map, precision = test(dataloader, model_save_path, args.sample_size)

print(ndcg)

run["eval"] = {
    "ndcg@1": ndcg[0],
    "ndcg@5": ndcg[1],
    "ndcg@10": ndcg[2],
    "map@1": _map[0],
    "map@5": _map[1],
    "map@10": _map[2],
    "recall@1": _map[0],
    "recall@5": _map[1],
    "recall@10": _map[2],
    "precision@1": precision[0],
    "precision@5": precision[1],
    "precision@10": precision[2],
}

run.stop()
