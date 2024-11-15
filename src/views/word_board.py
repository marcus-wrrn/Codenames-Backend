from enum import Enum
from src.database.orm import WordDatabase
import numpy as np
import random
from dataclasses import dataclass

class WordColor(Enum):
    RED = 1
    BLUE = 2
    GREY = 3
    BLACK = 4

class CustomBoards(Enum):
    ANIMALS = "animals"
    COUNTRIES = "countries"

@dataclass
class BoardWord:
    board_id: int
    database_id: int
    word: str
    color: WordColor
    active: bool = True
    
    def to_dict(self):
        return {
            'id': self.board_id,
            'word': self.word,
            'colorID': self.color.value,
            'active': self.active
        }

def get_enemy_team(player_team: WordColor) -> WordColor:
        return WordColor.RED if player_team == WordColor.BLUE else WordColor.BLUE

class Board:
    def __init__(self, word_objs: tuple=None, words: list[BoardWord]=None) -> None:
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
            
            words.append(BoardWord(i, word_id, word, color))
        
        # Randomize word order
        random.shuffle(words)
        return words

    def categorize_words_common(self, words: list[BoardWord], player_team: WordColor) -> tuple[dict[WordColor, list[BoardWord]], BoardWord]:
        enemy_team = get_enemy_team(player_team)
        
        categorized_words = {
            player_team: [],
            enemy_team: [],
            WordColor.GREY: [],
            WordColor.BLACK: []
        }

        for word in filter(lambda w: w.active, words):
            categorized_words.get(word.color, categorized_words[WordColor.GREY]).append(word)
        
        return categorized_words

    def categorize_words(self, player_team: WordColor) -> tuple[list[BoardWord], list[BoardWord], list[BoardWord], BoardWord]:
        """Categorizes words relative to the player's team"""
        categorized_words = self.categorize_words_common(self.words, player_team)
        return categorized_words[player_team], categorized_words[get_enemy_team(player_team)], categorized_words[WordColor.GREY], categorized_words[WordColor.BLACK][0]

    def map_categorized_embeddings(self, player_team: WordColor, board_embs: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Maps embeddings to categorized words relative to the player's team"""
        categorized_words = self.categorize_words_common(self.words, player_team)

        positive = [board_embs[word.database_id] for word in categorized_words[player_team]]
        negative = [board_embs[word.database_id] for word in categorized_words[get_enemy_team(player_team)]]
        neutral = [board_embs[word.database_id] for word in categorized_words[WordColor.GREY]]
        assassin = board_embs[categorized_words[WordColor.BLACK][0].database_id]

        return positive, negative, neutral, assassin

    def to_dict(self):
        return [word.to_dict() for word in self.words]
    
def init_gameboard(db_path: str):
    with WordDatabase(db_path) as db:
        word_objs = db.get_board()
    return Board(word_objs)

def create_custom_gameboard(db_path: str, custom_board: str):
    if custom_board not in CustomBoards.__members__:
        raise ValueError('Invalid custom board')
    
    with WordDatabase(db_path) as db:
        if custom_board == CustomBoards.COUNTRIES:
            return init_gameboard(db_path)
        elif custom_board == CustomBoards.ANIMALS:
            return db.get_animal_board()
    
    return init_gameboard(db_path)

def create_board_from_response(db_path: str, word_objs: list[tuple]) -> Board:
    if not word_objs:
        raise ValueError('Empty word objects')
    
    with WordDatabase(db_path) as db:
        words = []
        for word_obj in word_objs:
            key = word_obj['id']
            text = word_obj['word']
            color = WordColor(word_obj['colorID'])
            active = word_obj['active']

            # Find word_id in database
            word_id = db.get_word_id(text)
            words.append(BoardWord(key, word_id, text, color, active))
    
    return Board(words=words)