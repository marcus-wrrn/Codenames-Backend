from src.utils.word_board import Board, WordColor
from src.model import MORSpyFull
from src.reranker import ManyOutObj
from src.database.orm import Database
from src.database.vector_search import VectorSearch
import numpy as np
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

    def inference(self, board: Board, player_team: WordColor) -> ManyOutObj | None:
        try:
            board_embs = np.load(self.board_emb_path)
            pos_embs, neg_embs, neut_embs, assas_emb = board.map_categorized_embeddings(player_team, board_embs)

            pos_embs = torch.tensor(pos_embs, device=self.device).unsqueeze(0)
            neg_embs = torch.tensor(neg_embs, device=self.device).unsqueeze(0)
            neut_embs = torch.tensor(neut_embs, device=self.device).unsqueeze(0)
            assas_emb = torch.tensor(assas_emb, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.model(pos_embs, neg_embs, neut_embs, assas_emb)
            return logits
        except Exception as e:
            return None
        
    def get_hint_word(self, board: Board, player_team: WordColor) -> str:
        model_out = self.inference(board, player_team)
        
        if not model_out:
            return ''
        
        highest_scoring_index = torch.argmax(model_out.reranker_out)
        highest_scoring_word = model_out.texts[0][highest_scoring_index]
        return highest_scoring_word


    





    