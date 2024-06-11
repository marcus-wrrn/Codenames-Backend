from enum import Enum
from src.views.word_board import WordColor, Word
from src.database.orm import WordDatabase

class Team(Enum):
    PLAYER = 1
    BOT = 2

class GameLog:
    def __init__(self, log_data: list, db_path: str) -> None:
        try:
            with WordDatabase(db_path) as db:
                self.turns = [GameTurn(turn_data, db) for turn_data in log_data]
        except Exception as e:
            raise ValueError(f'Error processing game log: {str(e)}')
    
    def to_dict(self):
        return {
            'turns': [turn.to_dict() for turn in self.turns]
        }


class GameTurn:
    def __init__(self, turn_data, db: WordDatabase) -> None:
        if 'team' not in turn_data or 'chosenWords' not in turn_data or 'words' not in turn_data or 'hintInfo' not in turn_data:
            raise ValueError('Missing required fields in turn object')

        self.team = Team.PLAYER if turn_data['team'] == 1 else Team.BOT

        self.words = self._process_words(turn_data['words'], db)
        self.chosen_words = self._process_words(turn_data['chosenWords'], db, is_chosen=True)
        self.hint_word, self.sim_word_ids, self.sim_scores = self._process_hint_info(turn_data['hintInfo'])
    
    def _process_words(self, words: list[dict], db: WordDatabase, is_chosen=False) -> None:
        for word in words:
            if 'word' not in word or 'colorID' not in word or (not is_chosen and 'active' not in word):
                raise ValueError(f'Missing required fields in word object: {word}')
            database_id = db.get_word_id(word['word'])
            word['database_id'] = database_id
        return words

    def _process_hint_info(self, hint_info: dict):
        if 'hint_word' not in hint_info or 'sim_word_ids' not in hint_info or 'sim_scores' not in hint_info:
            raise ValueError('Missing required fields in hint info object')
        
        hint_word = hint_info['hint_word']
        sim_word_ids = hint_info['sim_word_ids']
        sim_scores = hint_info['sim_scores']

        return hint_word, sim_word_ids, sim_scores
    
    def to_dict(self):
        return {
            'team': self.team.value,
            'words': self.words,
            'chosenWords': self.chosen_words,
            'hint_info': {
                'hint_word': self.hint_word,
                'sim_word_ids': self.sim_word_ids,
                'sim_scores': self.sim_scores
            }
        }

    