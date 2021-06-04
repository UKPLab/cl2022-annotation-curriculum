import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ca.datasets import load_conll2003
from ca.difficulty.difficulty_estimators import StaticDifficultyEstimator


class LSTMTagger(nn.Module):
    # Simple lstm boilerplate (for testing)
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, label_size)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(len(sentence), 1, -1))
        label_out = self.output(lstm_out.view(len(sentence), -1))
        label_scores = F.log_softmax(label_out, dim=1)
        return label_scores


def get_random_embeddings(df: pd.DataFrame, embed_dim: int) -> dict:
    # Whacky, but just for testing
    embeddings = dict()
    for word in set(df["word"]):
        embeddings[word] = np.random.rand(embed_dim)
    return embeddings


def get_label_dict(df: pd.DataFrame) -> dict:
    labels = dict()
    for i, label in enumerate(sorted(set(df["ne"]))):
        labels[label] = i
    return labels


def main():
    # Simple usage examples for conll2003 corpus
    corpus = load_conll2003()

    diff = StaticDifficultyEstimator()
    data = corpus.train.head(1000)  # small test set

    for doc in data.groupby(["document_id"])["word"].apply(" ".join):
        print(doc, diff.get_flesch_kincaid(doc))
        print(doc, diff.get_bert_difficulty(1.1, doc))

    EMBEDDING_DIM = 100
    emb = get_random_embeddings(data, EMBEDDING_DIM)
    labels = get_label_dict(data)

    model = LSTMTagger(EMBEDDING_DIM, 50, len(emb), len(labels))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1):
        for doc, label in zip(
            data.groupby(["document_id"])["word"].apply(list), data.groupby(["document_id"])["ne"].apply(list)
        ):

            model.zero_grad()

            X = np.stack([emb[word] for word in doc])
            y = torch.tensor([labels[l] for l in label])

            predict = model(torch.from_numpy(X).float())
            print(" ".join(doc), diff.get_token_entropy(F.softmax(predict, dim=1).detach().numpy()))

            loss = loss_function(predict, y)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
