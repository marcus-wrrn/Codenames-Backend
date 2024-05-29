import torch
from src.model import MORSpyFull
from src.dataset import CodeNamesDataset
from src.vector_search import VectorSearch                                                                                                                                                                                                                                                                                                                                                                                                     

def init_model_and_vocab(model_path: str, vocab_dir: str, board_dir: str, device='cpu') -> tuple[MORSpyFull, VectorSearch]:
    dataset = CodeNamesDataset(vocab_dir, board_dir)
    vocab = VectorSearch(dataset)
    model = MORSpyFull(
        vocab=vocab, 
        device=device, 
        freeze_encoder=True
    )
    model.to(device)

    model.load_state_dict(torch.load(model_path))

    return model, vocab