from src.utils.model_loader import ModelLoader
from src.views.word_board import WordColor, init_gameboard
from env import DB_PATH


loader = ModelLoader()

board = init_gameboard(DB_PATH)

#loader.play_turn_algorithmic(board, WordColor.RED)

texts, scores, avg_score = loader.search_vocabulary(board_id=313, num_words=80)
texts_enc, scores_enc, avg_score_enc = loader.search_vocabulary(query='shakespeare', num_words=80)
print()