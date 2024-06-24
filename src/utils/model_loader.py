from src.views.word_board import Board, WordColor
from src.model import MORSpyFull, MORSpyManyPooled
from src.reranker import ManyOutObj, Reranker
from src.database.orm import WordDatabase
from src.database.vector_search import VectorSearch
from src.views.gameturn import GameTurn
import src.utils.utilities as utils
from env import MODEL_PATH, DB_PATH, VOCAB_EMB_PATH, BOARD_EMB_PATH, ENCODER_PATH
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import logging
from sentence_transformers import SentenceTransformer


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

        # with Database(db_path) as db:
        #     self.vocab = VectorSearch(db, self.vocab_emb_path)
        self.vocab = VectorSearch(index_path='./data/', load_from_index=True)
        #self.model = self._init_model(self.vocab, model_path)
        self.semantic_backbone = SentenceTransformer('all-mpnet-base-v2')
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

            pos_words, neg_words, neut_words, assas_word = board.categorize_words(player_team)
            
            # map words and embeddings
            all_words = pos_words + neg_words + neut_words + [assas_word]
            all_embs = torch.cat((pos_embs, neg_embs, neut_embs, assas_emb), dim=0)

            word_emb = logits.h_score_emb
            # Find the most similar words to the hint word
            sim_scores = F.cosine_similarity(word_emb, all_embs)
            scores, indices = sim_scores.sort(descending=True)
            indices = indices.cpu().numpy()
            board_ids = [all_words[i].board_id for i in indices]
        except Exception as e:
            logging.error(f'Error: {e}')
            return None
        
        return highest_scoring_word, board_ids, scores.cpu().numpy().tolist()
    
    def search_vocabulary(self, query: str, num_words=20) -> list[str]:
        # encode query
        query = self.semantic_backbone.encode(query, convert_to_tensor=True).unsqueeze(0)
        texts, embs, dist = self.vocab.search(query, num_results=num_words)
        texts = texts[0]
        embs = torch.tensor(embs, device=self.device).squeeze(0)
        scores = F.cosine_similarity(query, embs)
        avg_score = scores.mean().item()
        return texts.tolist(), scores.tolist(), avg_score
    
    def modify_embeddings(self, anchor: Tensor, embs: Tensor, score_vector: Tensor) -> np.ndarray:
        anchor = anchor.unsqueeze(0).repeat(embs.shape[0], 1)
        modify_amount = anchor - embs
        modify_amount = modify_amount * score_vector.unsqueeze(1)
        return F.normalize(modify_amount + embs, p=2, dim=1).cpu().numpy()

    def process_turn(self, turn: GameTurn):
        # Get expected turn data
        sim_scores = turn.sim_scores
        sim_ids = turn.sim_word_ids

        # Get player chosen data
        chosen_words = turn.chosen_words
        chosen_scores = utils.get_chosen_scores(chosen_words, sim_scores, sim_ids)

        # Find expected choices
        expected_words, expected_scores = utils.get_expected_choices_and_scores(turn.words, sim_scores, sim_ids, turn.team.value)

        # remove words from expected choices that have already been chosen
        expected_words, expected_scores = utils.prune_words(expected_words, expected_scores, chosen_words)
        if len(expected_words) == 0:
            return

        # Calculate modified scores
        chosen_modified = utils.adain(chosen_scores, expected_scores)
        expected_modified = utils.adain(expected_scores, chosen_scores, use_max=False)

        # Map embeddings
        board_embeddings = np.load(self.board_emb_path)
        chosen_embeddings = torch.tensor([board_embeddings[word.database_id] for word in chosen_words], device=self.device)
        expected_embeddings = torch.tensor([board_embeddings[word.database_id] for word in expected_words], device=self.device)

        vocab_id = np.where(self.vocab.vocab_texts == turn.hint_word)[0][0]
        hint_embedding = torch.tensor(self.vocab.vocab_embeddings[vocab_id], device=self.device)
        
        modify_chosen_amount = torch.tensor(chosen_modified - chosen_scores, device=self.device)
        modify_expected_amount = torch.tensor(expected_modified - expected_scores, device=self.device)

        modified_chosen_embs = self.modify_embeddings(hint_embedding, chosen_embeddings, modify_chosen_amount)
        modified_expected_embs = self.modify_embeddings(hint_embedding, expected_embeddings, modify_expected_amount)

        for i, word in enumerate(chosen_words):
            board_embeddings[word.database_id] = modified_chosen_embs[i]
        for i, word in enumerate(expected_words):
            board_embeddings[word.database_id] = modified_expected_embs[i]
        
        np.save(self.board_emb_path, board_embeddings)