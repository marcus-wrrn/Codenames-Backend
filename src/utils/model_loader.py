from src.utils.word_board import Board, WordColor
from src.model import MORSpyFull, MORSpyManyPooled
from src.reranker import ManyOutObj, Reranker
from src.database.orm import Database
from src.database.vector_search import VectorSearch
from env import MODEL_PATH, DB_PATH, VOCAB_EMB_PATH, BOARD_EMB_PATH, ENCODER_PATH
import numpy as np
import torch
import torch.nn.functional as F
import logging

class ModelLoader:
    def __init__(self, 
                 model_path=MODEL_PATH, 
                 encoder_path=ENCODER_PATH,
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

        self.encoder = MORSpyManyPooled()
        self.encoder.to(self.device)
        self.encoder.load_state_dict(torch.load(encoder_path))

        self.reranker = Reranker(self.vocab, neut_weight=2.0, device=self.device)
        
    
    def _init_model(self, vocab: VectorSearch, model_path: str) -> MORSpyFull:
        model = MORSpyFull(
            vocab=vocab,
            device=self.device,
            freeze_encoder=True
        )

        model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        return model

    def _get_embeddings(self, board: Board, player_team: WordColor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Extracts the embeddings for the board and maps them to the correct categories"""
        board_embs = np.load(self.board_emb_path)
        pos_embs, neg_embs, neut_embs, assas_emb = board.map_categorized_embeddings(player_team, board_embs)

        pos_embs = torch.tensor(pos_embs, device=self.device).unsqueeze(0)
        neg_embs = torch.tensor(neg_embs, device=self.device).unsqueeze(0)
        neut_embs = torch.tensor(neut_embs, device=self.device).unsqueeze(0)
        assas_emb = torch.tensor(assas_emb, device=self.device).unsqueeze(0)

        return pos_embs, neg_embs, neut_embs, assas_emb
    
    def _inference(self, pos_embs, neg_embs, neut_embs, assas_emb) -> ManyOutObj:
        with torch.no_grad():
            logits = self.model(pos_embs, neg_embs, neut_embs, assas_emb)
        return logits
    
    # def play_turn(self, board: Board, player_team: WordColor) -> tuple[str, list[int]] | None:
    #     """Play a turn and return the highest scoring word and the key values of the most similar words"""
    #     try:           
    #         pos_embs, neg_embs, neut_embs, assas_emb = self._get_embeddings(board, player_team)
    #         model_out = self._inference(pos_embs, neg_embs, neut_embs, assas_emb)
            
    #         # Get highest scoring word
    #         highest_scoring_index = torch.argmax(model_out.reranker_out)
    #         highest_scoring_word = model_out.texts[0][highest_scoring_index]

    #         # Map most similar words
    #         word_emb = model_out.h_score_emb

    #         pos_embs = pos_embs.squeeze(0)
    #         neg_embs = neg_embs.squeeze(0)
    #         neut_embs = neut_embs.squeeze(0)

    #         all_embs = torch.cat((pos_embs, neg_embs, neut_embs, assas_emb), dim=0)

    #         pos_words, neg_words, neut_words, assas_word = board.categorize_words(player_team)
    #         all_words = pos_words + neg_words + neut_words + [assas_word]

    #         # Find the most similar words to the hint word
    #         sim_scores = F.cosine_similarity(word_emb, all_embs)
    #         indices = sim_scores.sort(descending=True).indices
    #         indices = indices.cpu().numpy()
    #         word_keys = [all_words[i].key for i in indices]

    #         return highest_scoring_word, word_keys

    #     except Exception as e:
    #         logging.error(f'Error: {e}')
    #         return None
    
    def _encoder_inference(self, pos_embs, neg_embs, neut_embs, assas_emb) -> torch.Tensor:
        with torch.no_grad():
            encoder_logits, _ = self.encoder(pos_embs, neg_embs, neut_embs, assas_emb)
        return encoder_logits
    
    def play_turn_algorithmic(self, board: Board, player_team: WordColor) -> tuple[str, list[int]] | None:
        try:

            pos_embs, neg_embs, neut_embs, assas_emb = self._get_embeddings(board, player_team)

            encoder_logits = self._encoder_inference(pos_embs, neg_embs, neut_embs, assas_emb)
            
            logits = self.reranker.rerank_and_process(encoder_logits, pos_embs, neg_embs, neut_embs, assas_emb)

            pos_words, neg_words, neut_words, assas_word = board.categorize_words(player_team)
            all_words = pos_words + neg_words + neut_words + [assas_word]
            highest_scoring_index = logits.emb_ids.item()
            highest_scoring_word = logits.texts[0][highest_scoring_index]

            pos_embs = pos_embs.squeeze(0)
            neg_embs = neg_embs.squeeze(0)
            neut_embs = neut_embs.squeeze(0)

            all_embs = torch.cat((pos_embs, neg_embs, neut_embs, assas_emb), dim=0)

            pos_words, neg_words, neut_words, assas_word = board.categorize_words(player_team)
            all_words = pos_words + neg_words + neut_words + [assas_word]

            word_emb = logits.h_score_emb
            # Find the most similar words to the hint word
            sim_scores = F.cosine_similarity(word_emb, all_embs)
            indices = sim_scores.sort(descending=True).indices
            indices = indices.cpu().numpy()
            word_keys = [all_words[i].key for i in indices]
        except Exception as e:
            logging.error(f'Error: {e}')
            return None
        
        return highest_scoring_word, word_keys


        
    def get_hint_word(self, board: Board, player_team: WordColor) -> str:
        model_out = self.play_turn(board, player_team)

        if not model_out:
            return ''
        
        highest_scoring_index = torch.argmax(model_out.reranker_out)
        highest_scoring_word = model_out.texts[0][highest_scoring_index]


        return highest_scoring_word





    