from enum import Enum
from src.database.orm import Database
import random

class WordColor(Enum):
    RED = 1
    BLUE = 2
    GREY = 3
    BLACK = 4


class Word:
    def __init__(self, key: int, word_id: int, word: str, color: WordColor, active=True) -> None:
        self.key = key
        self.id = word_id
        self.word = word
        self.color = color
        self.active = active
    
    def to_dict(self):
        return {
            'id': self.key,
            'word': self.word,
            'colorID': self.color.value,
            'active': self.active
        }

class Board:
    def __init__(self, word_objs: tuple, words: list[Word]=None) -> None:
        self.words = None
        if words:
            self.words = words
        else:
            self.words = self._init_board(word_objs)
        
    def _init_board(self, word_objs: tuple):
        words = []
        for i, word_obj in enumerate(word_objs):
            word = word_obj[0]
            word_id = word_obj[1]

            color = WordColor.GREY
            if i < 9:
                color = WordColor.RED
            elif i < 18:
                color = WordColor.BLUE
            elif i == len(word_objs) - 1:
                color = WordColor.BLACK
            
            words.append(Word(i, word_id, word, color))
        
        # Randomize word order
        random.shuffle(words)
        return words

    def categorize_words(self, 
            player_team: WordColor) -> tuple[list[Word], list[Word], list[Word], Word]:
        """Categorizes words relative to the players team"""
        positive = []
        negative = []
        neutral = []
        assassin = None
        for i, word in self.words:
            if not word.active: continue

            if word.color == player_team:
                positive.append(word)
            elif word.color == WordColor.GREY:
                neutral.append(word)
            elif word.color == WordColor.BLACK:
                assassin = word
            else:
                negative.append(word)
        
        return positive, negative, neutral, assassin



    def to_dict(self):
        return [word.to_dict() for word in self.words]
    
def init_gameboard(db_path: str):
    with Database(db_path) as db:
        word_objs = db.get_board()
    return Board(word_objs)