from dataloaders.QADataModule import QADataModule
from sentence_transformers import SentenceTransformer, losses, models
import os


def train_bi_encoder():
    data_module = QADataModule(data_dir='./data')
    data_module.setup()
    train_dataloader = data_module.train_dataloader()

    # Now we create a SentenceTransformer model from scratch
    word_emb = models.Transformer('sentence-transformers/msmarco-distilbert-base-dot-prod-v3')
    pooling = models.Pooling(word_emb.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_emb, pooling])

    # MultipleNegativesRankingLoss requires input pairs (query, relevant_passage)
    # and trains the model so that is is suitable for semantic search
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Tune the model
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps,
              show_progress_bar=True)

    os.makedirs('models', exist_ok=True)
    os.makedirs('models/bi-encoders', exist_ok=True)
    model.save('models/bi-encoders/bi-encoder-v1')


if __name__ == "__main__":
    train_bi_encoder()
