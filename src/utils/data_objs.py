from enum import Enum

class WordColor(Enum):
    RED = 1
    BLUE = 2
    GREY = 3
    BLACK = 4


class Word:
    def __init__(self, id: int, word: str, color: WordColor, active=True) -> None:
        self.id = id
        self.word = word
        self.color = color
        self.active = active
    
    def to_dict(self):
        return {
            'id': self.id,
            'word': self.word,
            'color': self.color.value,
            'active': self.active
        }