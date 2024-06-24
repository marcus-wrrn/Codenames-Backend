from src.utils.process_turn import process_turn
from src.views.gameturn import GameTurn
from src.test_data import test_turn_model_choice, test_turn_player_choice
from src.database.orm import WordDatabase
from src.utils.model_loader import ModelLoader
from env import DB_PATH, BOARD_EMB_PATH


if __name__ == '__main__':
    with WordDatabase(DB_PATH) as db:
        turn = GameTurn(test_turn_player_choice['past_turn'], db)
    loader = ModelLoader()
    loader.process_turn(turn)