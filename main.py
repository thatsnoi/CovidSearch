import argparse
from utils import download_dataset
from query_gen_and_train import gen_queries, train_bi_encoder
from test import test
import neptune.new as neptune
from os import path
from zipfile import ZipFile

parser = argparse.ArgumentParser(description='Parameters for TREC-Covid IR.')
parser.add_argument('--name', type=str, help='Name of the experiment')

# Only use subset (for testing)
parser.add_argument('--sample_size', type=int, default=None, help='Corpus sample size')

# Bi-encoder fine-tuning
parser.add_argument('--pretrained_model', type=str, default='dmis-lab/biobert-v1.1',
                    help='Pretrained huggingface model for bi-encoder training.')
parser.add_argument('--num_epochs', type=int, default=10, help='Epochs for bi-encoder training.')
parser.add_argument('--model_name', type=str)
parser.add_argument('--questions_subset', type=float, default=0.1)

# Use pre-generated queries
parser.add_argument('--gen', type=str, default=None, help='Generated query files from neptune')

# Use pre-fine-tuned bi-encoder
parser.add_argument('--bi_encoder_neptune_id', type=str, default=None, help='Neptune experiment id for bi-encoder')
parser.add_argument('--bi_encoder_path', type=str, default='./output/biobert-GenQ-bi-encoder')

# Score Function
parser.add_argument('--score_function', type=str, default='dot')

# Cross encoder
parser.add_argument('--cross_encoder', action='store_true')
parser.set_defaults(cross_encoder=False)
parser.add_argument('--cross_encoder_top_k', type=int, default=20)

# Fuse with BM25
parser.add_argument('--fuse_with_bm25', action='store_true')
parser.set_defaults(fuse_with_bm25=False)

args = parser.parse_args()

dataloader, data_path = download_dataset()

run = neptune.init(
    project="noahjadallah/TREC-Covid",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzgwNzhlNy1iMDVlLTQwNWUtYWJlYS04NWMxNjA0YmQ3ODAifQ==",
)

params = {
    "pretrained_model": args.pretrained_model,
    "bi-encoder_model": args.model_name,
    "sample_size": args.sample_size,
    "num_epochs": args.num_epochs,
    "score_function": args.score_function,
    "fuse_with_bm25": args.fuse_with_bm25,
    "questions_subset": args.questions_subset,
    "cross_encoder": args.cross_encoder,
    "cross_encoder_top_k": args.cross_encoder_top_k
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

    model_save_path = train_bi_encoder(data_path, num_epochs=args.num_epochs, pretained_model=args.pretrained_model,
                                       model_name=args.model_name, subset_size=args.questions_subset)

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

ndcg, _map, recall, precision = test(dataloader, model_save_path, args.sample_size, score_function=args.score_function,
                                     fuse_with_bm25=args.fuse_with_bm25, cross_encoder_top_k=args.cross_encoder_top_k)

run["eval/ndcg"] = ndcg
run["eval/map"] = _map
run["eval/recall"] = recall
run["eval/precision"] = precision

run.stop()
