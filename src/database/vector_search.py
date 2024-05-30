import numpy as np
from src.database.orm import Database
import faiss
import torch

class VectorSearch:
    def __init__(self, db: Database, vocab_path: str, n_dim=768, n_neighbours=32) -> None:
        vocab_words = db.get_pruned_vocab()
        
        unpruned_embeddings = np.load(vocab_path)
        pruned_embeddings = []
        texts = []
        for word in vocab_words:
            texts.append(word[0])
            pruned_embeddings.append(unpruned_embeddings[word[1]])
        
        self.vocab_texts = np.array(texts)
        self.vocab_embeddings = np.array(pruned_embeddings, dtype=np.float32)

        # Initialize index + add embeddings
        self.index = faiss.IndexHNSWFlat(n_dim, n_neighbours)
        self.index.add(self.vocab_embeddings)

    def vocab_add(self, text: str, emb: torch.Tensor):
        emb.detach().cpu().numpy()
        self.index.add(emb)
        self.vocab_embeddings.append(emb)
        self.vocab_words(text)
    
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
    
    def save_index(self, filepath: str):
        faiss.write_index(self.index, filepath)
    
