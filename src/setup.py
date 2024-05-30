import torch
from src.model import MORSpyFull
from src.dataset import CodeNamesDataset
from database.vector_search import VectorSearch                                                                                                                                                                                                                                                                                                                                                                                                     

def init_model_and_vocab(model_path: str, dataset: CodeNamesDataset, device='cpu') -> tuple[MORSpyFull, VectorSearch]:
    # Initialize data
    vocab = VectorSearch(dataset)

    # Initialize model
    model = MORSpyFull(
        vocab=vocab, 
        device=device, 
        freeze_encoder=True
    )
    # Load model
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    return model, vocab