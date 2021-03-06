import torch
from torch.utils.data import Dataset
import numpy as np


class CTMDataset(Dataset):

    """Class to load BOW dataset."""

    def __init__(self, X, X_bert, idx2token):
        """
        Args
            X : array-like, shape=(n_samples, n_features)
                Document word matrix.
        """
        if len(X) != len(X_bert):
            raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                            "You might want to check if the BoW preparation method has removed some documents. ")

        self.X = X
        self.X_bert = X_bert
        self.idx2token = idx2token

    def __len__(self):
        """Return length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        X = torch.FloatTensor(self.X[i])
        X_bert = torch.FloatTensor(self.X_bert[i])

        return {'X': X, 'X_bert': X_bert}


class CTMDatasetTopReg(Dataset):

    """Class to load BOW dataset."""

    def __init__(self, X, X_bert, idx2token, labels):
        """
        Args
            X : array-like, shape=(n_samples, n_features)
                Document word matrix.
        """
        if len(X) != len(X_bert):
            raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                            "You might want to check if the BoW preparation method has removed some documents. ")
        if len(X) != len(labels):
            raise Exception("Text and labels have different sizes!")

        self.X = X
        self.X_bert = X_bert
        self.idx2token = idx2token
        self.labels = labels

    def __len__(self):
        """Return length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        X = torch.FloatTensor(self.X[i])
        X_bert = torch.FloatTensor(self.X_bert[i])
        label = torch.LongTensor([self.labels[i]])

        return {'X': X, 'X_bert': X_bert, 'label': label}


