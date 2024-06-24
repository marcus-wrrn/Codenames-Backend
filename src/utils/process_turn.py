from src.views.gameturn import GameTurn, TurnWord
from src.database.orm import WordDatabase
import numpy as np
import torch
import torch.nn.functional as F
from env import BOARD_EMB_PATH, DB_PATH, VOCAB_EMB_PATH

def adain(x: np.ndarray, y: np.ndarray, use_max=True) -> np.ndarray:
    """Adaptive instance normalization between two numpy arrays"""
    if x.size == 0 or y.size == 0:
        return x.copy()
    # To avoid division by zero
    div_zero_offset = 0.0000000000000001
    # Calculate the mean and standard deviation of x and y
    std_x = x.std() + div_zero_offset
    std_y = y.std() + div_zero_offset
    mean_x = x.mean()
    mean_y = y.max() if use_max else y.min()
    return ((x - mean_x) / std_x) * std_y + mean_y

def get_expected_choices_and_scores(words: list[TurnWord], sim_scores: list[float], sim_ids: list[int], team_value: int):
    ordered_words = sorted(words, key=lambda x: x.id)
    ordered_words = [ordered_words[i] for i in sim_ids]
    expected_scores = []
    expected_words = []
    for i, word in enumerate(ordered_words):
        expected_scores.append(sim_scores[i])
        expected_words.append(word)
        if (word.colorID != team_value):
            break
    return expected_words, expected_scores

def get_chosen_scores(chosen_words: list[TurnWord], sim_scores: list[float], sim_ids: list[int]):
    chosen_ids = [word.id for word in chosen_words]
    sim_id_to_score = {sim_id: score for sim_id, score in zip(sim_ids, sim_scores)}
    chosen_scores = [sim_id_to_score[id] for id in chosen_ids if id in sim_id_to_score]

    return chosen_scores

def prune_words(expected_words, expected_scores, chosen_words):
    # Remove words from expected choices that have already been chosen
    chosen_ids = {word.id for word in chosen_words}  # Use a set for faster lookup

    pruned_words = []
    pruned_scores = []

    for word, score in zip(expected_words, expected_scores):
        if word.id not in chosen_ids:
            pruned_words.append(word)
            pruned_scores.append(score)

    return pruned_words, pruned_scores

def get_hint_embedding(hint_word: str, vocab_embedding_path: str = VOCAB_EMB_PATH, db_path = DB_PATH):
    with WordDatabase(db_path) as db:
        hint_id = db.get_word_id(hint_word, from_board=False)
    vocab_embeddings = np.load(vocab_embedding_path)
    return torch.tensor(vocab_embeddings[hint_id])


def process_turn(turn: GameTurn, board_embedding_path: str = BOARD_EMB_PATH):
    # Get expected turn data
    sim_scores = turn.sim_scores
    sim_ids = turn.sim_word_ids

    # Get actual turn data
    chosen_words = turn.chosen_words
    chosen_scores = get_chosen_scores(chosen_words, sim_scores, sim_ids)

    # Find expected choices
    expected_words, expected_scores = get_expected_choices_and_scores(turn.words, sim_scores, sim_ids, turn.team.value)

    # remove words from expected choices that have already been chosen
    expected_words, expected_scores = prune_words(expected_words, expected_scores, chosen_words)

    chosen_scores = np.array(chosen_scores)
    expected_scores = np.array(expected_scores)

    chosen_modified = adain(chosen_scores, expected_scores)
    expected_modified = adain(expected_scores, chosen_scores, use_max=False)

    # Map embeddings

    board_embeddings = np.load(board_embedding_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chosen_embeddings = torch.tensor([board_embeddings[word.database_id] for word in chosen_words], device=device)
    expected_embeddings = torch.tensor([board_embeddings[word.database_id] for word in expected_words], device=device)
    hint_embedding = get_hint_embedding(turn.hint_word).to(device)

    chosen_similarity = F.cosine_similarity(chosen_embeddings, hint_embedding)
    expected_similarity = F.cosine_similarity(expected_embeddings, hint_embedding)
    print()
