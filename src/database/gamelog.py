from pymongo import MongoClient
from src.views.gameturn import GameLog
import uuid

class GameLogDatabase:
    """Used to record all past games between the model and players"""
    def __init__(self, db_path='localhost', port=27017, db_name='game_log_db', collection_name='logs'):
        self.client = MongoClient(db_path, port)
        self.db = self.client[db_name]
        self.logs = self.db[collection_name]

    def save_log(self, log: GameLog, game_words: list, word_colorIDs: list[int], origin_ip: str) -> bool:
        data = {
            'game_id': str(uuid.uuid4()),
            'log': log.to_dict(),
            'game_words': game_words,
            'word_colorIDs': word_colorIDs,
            'origin_ip': origin_ip # origin ip is used as a simple way of determining same/different players, for safety reasons this should be hashed
        }
        self.logs.insert_one(data)
        return True

    def close(self):
        self.client.close()

    def __enter__(self) -> 'GameLogDatabase':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            print(f"Exception has been handled: {exc_type}, {exc_val}")
        return False