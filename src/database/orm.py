import sqlite3

class Database:
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
    
    def get_board(self):
        self.cursor.execute('''
            SELECT word, word_id FROM board
            ORDER BY RANDOM()
            LIMIT 25;
        ''')
        return self.cursor.fetchall()
    
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


    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.cursor.close()
        self.conn.close()