import logging
import os
import pathlib
import random

from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses, models

# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def gen_queries(dataloader, data_path, sample_size=None, model_path="BeIR/query-gen-msmarco-t5-base-v1", ques_per_passage=5):
    corpus = dataloader.load_corpus()

    if sample_size is not None:
        random.seed(55)
        corpus = dict(random.sample(corpus.items(), sample_size))

    # question-generation model loading
    generator = QGen(model=QGenModel(model_path))

    # Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
    # https://huggingface.co/blog/how-to-generate

    # Generate queries per passage from docs in corpus and save them in data_path
    generator.generate(corpus, output_dir=data_path, ques_per_passage=ques_per_passage, prefix="gen")


def train_bi_encoder(data_path, model_name="dmis-lab/biobert-v1.1", model_path="biobert", num_epochs=10):
    # Training on Generated Queries
    corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix="gen").load(split="train")

    # Provide any HuggingFace model and fine-tune from scratch
    if model_name is not None:
        word_embedding_model = models.Transformer(model_name, max_seq_length=350)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        # Or provide already fine-tuned sentence-transformer model
        model = SentenceTransformer("msmarco-distilbert-base-v3")

    # Provide any sentence-transformers model path
    retriever = TrainRetriever(model=model, batch_size=32)

    # Prepare training samples
    train_samples = retriever.load_train(corpus, gen_queries, gen_qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

    # If no dev set is present evaluate using dummy evaluator
    ir_evaluator = retriever.load_dummy_evaluator()

    # Provide model save path
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output",
                                   "{}-GenQ-bi-encoder".format(model_path))
    os.makedirs(model_save_path, exist_ok=True)

    # Configure Train params
    evaluation_steps = 5000
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

    retriever.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=ir_evaluator,
                  epochs=num_epochs,
                  output_path=model_save_path,
                  warmup_steps=warmup_steps,
                  evaluation_steps=evaluation_steps,
                  use_amp=True)

    return model_save_path
