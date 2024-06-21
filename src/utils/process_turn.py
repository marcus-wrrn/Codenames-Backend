from src.views.gameturn import GameTurn


def process_turn(turn: GameTurn):
    # Get expected turn data
    sim_scores = turn.sim_scores
    sim_ids = turn.sim_word_ids
    words = turn.words

    # Order words based on sim_ids
    ordered_words = ...
    # Get actual turn data
    ...