from enum import Enum
from src.views.word_board import WordColor, Word
from src.database.orm import Database

class Team(Enum):
    PLAYER = 1
    BOT = 2

class GameLog:
    def __init__(self, log_data: list, db_path: str) -> None:
        try:
            with Database(db_path) as db:
                self.turns = [GameTurn(turn_data, db) for turn_data in log_data]
        except Exception as e:
            raise ValueError(f'Error processing game log: {str(e)}')
    
    def to_dict(self):
        return {
            'turns': [turn.to_dict() for turn in self.turns]
        }


class GameTurn:
    def __init__(self, turn_data, db: Database) -> None:
        if 'team' not in turn_data or 'chosenWords' not in turn_data or 'words' not in turn_data or 'hintInfo' not in turn_data:
            raise ValueError('Missing required fields')

        self.team = Team.PLAYER if turn_data['team'] == 1 else Team.BOT

        self.words = self.process_words(turn_data['words'], db)
        self.chosen_words = self.process_words(turn_data['chosenWords'], db)
        self.hint_word, self.sim_word_ids, self.sim_scores = self.process_hint_info(turn_data['hintInfo'])
    
    def process_words(self, words: list[dict], db: Database) -> None:
        # get word ids
        word_objs = []
        for word in words:
            if 'id' not in word or 'word' not in word or 'colorID' not in word or 'active' not in word:
                raise ValueError('Missing required fields')
            
            database_id = db.get_word_id(word['word'])
            word_objs.append(Word(word['id'], database_id, word['word'], WordColor(word['colorID']), word['active']))
        
        return word_objs

    def process_hint_info(self, hint_info: dict):
        if 'hint_word' not in hint_info or 'sim_word_ids' not in hint_info or 'sim_scores' not in hint_info:
            raise ValueError('Missing required fields')
        
        hint_word = hint_info['hint_word']
        sim_word_ids = hint_info['sim_word_ids']
        sim_scores = hint_info['sim_scores']

        return hint_word, sim_word_ids, sim_scores
    
    def to_dict(self):
        return {
            'team': self.team.value,
            'words': [word.to_dict() for word in self.words],
            'chosenWords': [word.to_dict() for word in self.chosen_words],
            'hint_info': {
                'hint_word': self.hint_word,
                'sim_word_ids': self.sim_word_ids,
                'sim_scores': self.sim_scores
            }
        }

    
