import sqlite3

class WordDatabase:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_table()

    def _init_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocab (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT,
                word_id INTEGER UNIQUE,
                is_board_word BOOLEAN DEFAULT 0
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS board (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT,
                word_id INTEGER UNIQUE
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS animal_board (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT,
                word_id INTEGER UNIQUE
            );
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS bad_words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT,
                word_id INTEGER UNIQUE
            );
        ''')

        self.conn.commit()
    
    def insert_vocab(self, word_id: int, word: str, commit=True) -> None:
        # Check if word exists in board
        self.cursor.execute('''
            SELECT word_id FROM board
            WHERE word = ?
        ''', (word,))

        is_board_word = False
        if self.cursor.fetchone():
            is_board_word = True
        
        self.cursor.execute('''
            INSERT INTO vocab (word_id, word, is_board_word)
            VALUES (?, ?, ?)
        ''', (word_id, word, is_board_word))

        if commit: self.conn.commit()
    
    def insert_board(self, word_id: int, word: str, commit=True) -> None:
        self.cursor.execute('''
            INSERT INTO board (word_id, word)
            VALUES (?, ?)
        ''', (word_id, word))
        if commit: self.conn.commit()
    
    def insert_animal_board(self, word_id: int, word: str, commit=True) -> None:
        self.cursor.execute('''
            INSERT INTO animal_board (word_id, word)
            VALUES (?, ?)
        ''', (word_id, word))
        if commit: self.conn.commit()
    
    def insert_bad_word(self, word_id: int, word: str, commit=True) -> None:
        self.cursor.execute('''
            INSERT INTO bad_words (word_id, word)
            VALUES (?, ?)
        ''', (word_id, word))
        if commit: self.conn.commit()
    
    def _get_board(self, board_name: str):
        query = f'''
            SELECT word, word_id FROM {board_name}
            ORDER BY RANDOM()
            LIMIT 25;
        '''
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_board(self):
        return self._get_board('board')

    def get_animal_board(self):
        return self._get_board('animal_board')
    
    def get_random_bad_words(self, num=15):
        self.cursor.execute('''
            SELECT word, word_id FROM bad_words
            ORDER BY RANDOM()
            LIMIT ?;
        ''', (num,))
        return self.cursor.fetchall()
    
    def get_bad_word(self, word: str):
        self.cursor.execute('''
            SELECT word_id FROM bad_words
            WHERE word = ?
        ''', (word,))
        return self.cursor.fetchone()
    
    def get_pruned_vocab(self):
        self.cursor.execute('''
            SELECT word, word_id FROM vocab
            WHERE is_board_word = 0
        ''')
        return self.cursor.fetchall()
    
    def get_word_id(self, word: str, from_board=True):
        # Ensure the table name is safe and valid
        table = 'board' if from_board else 'vocab'
        query = f'''
            SELECT word_id FROM {table}
            WHERE word = ?
        '''
        self.cursor.execute(query, (word,))
        return self.cursor.fetchone()[0]

    def get_all_board_words(self):
        self.cursor.execute('''
            SELECT word, word_id FROM board
        ''')
        return self.cursor.fetchall()
    
    def get_all_vocab_words(self):
        self.cursor.execute('''
            SELECT word, word_id FROM vocab
        ''')
        return self.cursor.fetchall()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.cursor.close()
        self.conn.close()