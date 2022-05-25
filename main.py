import argparse
from utils import download_dataset
from query_gen_and_train import gen_queries, train_bi_encoder
from test import test
import neptune.new as neptune
from os import path
from zipfile import ZipFile

parser = argparse.ArgumentParser(description='Parameters for TREC-Covid IR.')
parser.add_argument('--name',               type=str, help='Experiment name for logging')
parser.add_argument('--sample_size',        type=int, default=None, help='Corpus sample size')
parser.add_argument('--pretrained_model',   type=str, default='dmis-lab/biobert-v1.1',
                    help='Pretrained huggingface model for bi-encoder training.')
parser.add_argument('--num_epochs',         type=int, default=10, help='Epochs for bi-encoder training.')
parser.add_argument('--gen', type=str, default=None, help='Generated query files from neptune')
parser.add_argument('--bi_encoder', type=str, default=None, help='Bi-Encoder from neptune')
parser.add_argument('--bi_encoder_path', type=str, default='./output/biobert-GenQ-bi-encoder')

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


if args.bi_encoder is None:

    if args.gen is None:
        gen_queries(dataloader, data_path, sample_size=args.sample_size)
    else:
        print(f"Downloading generated queries from neptune project {args.gen}")
        project = neptune.init(
            project="noahjadallah/TREC-Covid",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzgwNzhlNy1iMDVlLTQwNWUtYWJlYS04NWMxNjA0YmQ3ODAifQ==",
            run=args.bi_encoder,
            mode="read-only"
        )
        project["dataset/gen/gen_queries.jsonl"].download(path.join(data_path, "gen-queries.jsonl"))
        project["dataset/gen/train.tsv"].download(path.join(data_path, "gen-qrels/train.tsv"))
        project.stop()

    run["dataset/gen/gen_queries.jsonl"].upload(path.join(data_path, "gen-queries.jsonl"))
    run["dataset/gen/train.tsv"].upload(path.join(data_path, "gen-qrels/train.tsv"))

    model_save_path = train_bi_encoder(data_path, num_epochs=args.num_epochs, model_name=args.pretrained_model)

else:
    print(f"Downloading bi-encoder from neptune project {args.bi_encoder}")
    project = neptune.init(
        project="noahjadallah/TREC-Covid",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzgwNzhlNy1iMDVlLTQwNWUtYWJlYS04NWMxNjA0YmQ3ODAifQ==",
        run=args.bi_encoder,
        mode="read-only"
    )
    project["model/bi-encoder"].download()
    project.stop()

    with ZipFile('bi-encoder.zip', 'r') as zipObj:
        zipObj.extractall()

    model_save_path = args.bi_encoder_path


run["model/bi-encoder"].upload_files(model_save_path)

ndcg, _map, recall, precision = test(dataloader, model_save_path, args.sample_size)

run["eval/ndcg"] = ndcg
run["eval/map"] = _map
run["eval/recall"] = recall
run["eval/precision"] = precision

run.stop()
