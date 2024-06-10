import numpy as np
from src.database.orm import Database
import faiss
import torch

class VectorSearch:
    def __init__(self, db: Database=None, vocab_path: str=None, n_dim=768, n_neighbours=32, load_from_index=False, index_path=None) -> None:
        if load_from_index:
            self.load_index(index_path)
            return
        vocab_words = db.get_pruned_vocab()
        
        # load vocab embeddings and words
        self.vocab_embeddings = np.load(vocab_path)
        vocab_words = db.get_pruned_vocab()

        # Initialize vocab words + embeddings
        self.vocab_texts = np.array([word[0] for word in vocab_words])

        # Initialize index + add embeddings
        self.index = faiss.IndexHNSWFlat(n_dim, n_neighbours)
        self.index.add(self.vocab_embeddings)

    def vocab_add(self, text: str, emb: torch.Tensor):
        emb.detach().cpu().numpy()
        self.index.add(emb)
        np.append(self.vocab_embeddings, emb)
        self.vocab_texts(text)
    
    def search(self, logits: torch.Tensor, num_results=20):
        # detach tensor from device and convert it to numpy for faiss compatibility
        search_input = logits.detach().cpu().numpy()
        # D: L2 distance from input, I: index of result
        D, I = self.index.search(search_input, num_results)
        # Map index values to words
        words = self.vocab_texts[I]
        embeddings = self.vocab_embeddings[I]
        return words, embeddings, D

    def index_to_texts(self, index):
        return self.vocab_texts[index]
    
    def save_index(self, filedir: str):
        index_path = filedir + 'index'
        faiss.write_index(self.index, index_path)
        vocab_path = filedir + 'vocab.npy'
        np.save(vocab_path, self.vocab_embeddings)

    def load_index(self, filedir: str):
        index_path = filedir + 'index'
        self.index = faiss.read_index(index_path)
        vocab_path = filedir + 'vocab.npy'
        self.vocab_embeddings = np.load(vocab_path)
        text_path = filedir + 'vocab_texts.npy'
        self.vocab_texts = np.load(text_path)
    
