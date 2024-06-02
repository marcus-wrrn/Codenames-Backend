from src.utils.word_board import Board, WordColor
from src.model import MORSpyFull
from src.reranker import ManyOutObj
from src.database.orm import Database
from src.database.vector_search import VectorSearch
from env import MODEL_PATH, DB_PATH, VOCAB_EMB_PATH, BOARD_EMB_PATH
import numpy as np
import torch
import torch.nn.functional as F
import logging

class ModelLoader:
    def __init__(self, 
                 model_path=MODEL_PATH, 
                 db_path=DB_PATH, 
                 vocab_emb_path=VOCAB_EMB_PATH, 
                 board_emb_path=BOARD_EMB_PATH) -> None:
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

    
    
    def _inference(self, pos_embs, neg_embs, neut_embs, assas_emb) -> ManyOutObj:
        with torch.no_grad():
            logits = self.model(pos_embs, neg_embs, neut_embs, assas_emb)
        return logits
    
    def play_turn(self, board: Board, player_team: WordColor) -> tuple[str, list[int]] | None:
        """Play a turn and return the highest scoring word and the key values of the most similar words"""
        try:
            board_embs = np.load(self.board_emb_path)
            pos_embs, neg_embs, neut_embs, assas_emb = board.map_categorized_embeddings(player_team, board_embs)
            pos_words, neg_words, neut_words, assas_word = board.categorize_words(player_team)

            pos_embs = torch.tensor(pos_embs, device=self.device).unsqueeze(0)
            neg_embs = torch.tensor(neg_embs, device=self.device).unsqueeze(0)
            neut_embs = torch.tensor(neut_embs, device=self.device).unsqueeze(0)
            assas_emb = torch.tensor(assas_emb, device=self.device).unsqueeze(0)
            
            model_out = self._inference(pos_embs, neg_embs, neut_embs, assas_emb)
            
            # Get highest scoring word
            highest_scoring_index = torch.argmax(model_out.reranker_out)
            highest_scoring_word = model_out.texts[0][highest_scoring_index]

            # Map most similar words
            word_emb = model_out.h_score_emb

            pos_embs = pos_embs.squeeze(0)
            neg_embs = neg_embs.squeeze(0)
            neut_embs = neut_embs.squeeze(0)

            all_embs = torch.cat((pos_embs, neg_embs, neut_embs, assas_emb), dim=0)
            all_words = pos_words + neg_words + neut_words + [assas_word]

            # Find the most similar words to the hint word
            sim_scores = F.cosine_similarity(word_emb, all_embs)
            indices = sim_scores.sort(descending=True).indices
            indices = indices.cpu().numpy()
            word_keys = [all_words[i].key for i in indices]

            return highest_scoring_word, word_keys

        except Exception as e:
            logging.error(f'Error: {e}')
            return None
        
    def get_hint_word(self, board: Board, player_team: WordColor) -> str:
        model_out = self.play_turn(board, player_team)

        if not model_out:
            return ''
        
        highest_scoring_index = torch.argmax(model_out.reranker_out)
        highest_scoring_word = model_out.texts[0][highest_scoring_index]


        return highest_scoring_word


    





    