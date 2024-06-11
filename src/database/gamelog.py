from pymongo import MongoClient
from src.views.gameturn import GameLog
import uuid

class GameLogDatabase:
    def __init__(self, db_path, port=27017, db_name='game_log_db', collection_name='logs'):
        

        self.client = MongoClient(db_path, port)
        self.db = self.client[db_name]
        self.logs = self.db[collection_name]

        self.db_path = db_path
        self.db_name = db_name
        self.collection_name = collection_name

    def save_log(self, log: GameLog, game_words: list[str], word_colorIDs: list[int], origin_ip: str):
        data = {
            'game_id': str(uuid.uuid4()),
            'log': log.to_dict(),
            'game_words': game_words,
            'word_colorIDs': word_colorIDs,
            'origin_ip': origin_ip
        }
        
        self.logs.insert_one(data)
        self.client.close()
        return True