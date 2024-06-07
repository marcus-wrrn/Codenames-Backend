from src.utils.model_loader import ModelLoader
from src.utils.word_board import WordColor, init_gameboard
from env import DB_PATH


loader = ModelLoader()

board = init_gameboard(DB_PATH)

loader.play_turn_algorithmic(board, WordColor.RED)