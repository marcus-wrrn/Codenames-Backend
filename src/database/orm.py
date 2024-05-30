import sqlite3


class ORM:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_table()

    def _init_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocab (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word_id INTEGER,
                word TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS board (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word_id INTEGER,
                word TEXT
            )
        ''')
        self.conn.commit()
    
    def insert_vocab(self, word_id: int, word: str) -> None:
        self.cursor.execute('''
            INSERT INTO vocab (word_id, word)
            VALUES (?, ?)
        ''', (word_id, word))
        self.conn.commit()
    
    def insert_board(self, word_id: int, word: str) -> None:
        self.cursor.execute('''
            INSERT INTO board (word_id, word)
            VALUES (?, ?)
        ''', (word_id, word))
        self.conn.commit()
    
    def get_board(self):
        self.cursor.execute('''
            SELECT word FROM board
            ORDER BY RANDOM()
            LIMIT 25;
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