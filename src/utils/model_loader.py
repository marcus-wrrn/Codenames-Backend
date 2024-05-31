from src.utils.word_board import Board, WordColor
from src.model import MORSpyFull
from src.reranker import ManyOutObj
from src.database.orm import Database
from src.database.vector_search import VectorSearch
import torch

class ModelLoader:
    def __init__(self, model_path: str, db_path: str, vocab_emb_path: str, board_emb_path: str) -> None:
        # Set the paths for the database and vocabulary
        self.db_path = db_path
        self.vocab_emb_path = vocab_emb_path
        self.board_emb_path = board_emb_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with Database(db_path) as db:
            self.vocab = VectorSearch(db, self.vocab_emb_path)
        
        self.model = self._init_model(self.vocab, model_path)
        
    
    def _init_model(self, vocab: VectorSearch, model_path: str) -> MORSpyFull:
        model = MORSpyFull(
            vocab=vocab,
            device=self.device,
            freeze_encoder=True
        )

        model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        return model

def inference(board: Board, player_team: WordColor) -> ManyOutObj:
    pos_words, neg_words, neut_words, assas_word = board.categorize_words(player_team)


    





    