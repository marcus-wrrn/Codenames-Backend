import unittest
import sys
import os
import numpy as np

# Adjust the path to ensure the src module can be imported
# TODO: Change directory structure so that this is not necessary
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.orm import WordDatabase
import env

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Yes I'm using the same db path, should not be an issue as a new db can be generated rapidly
        self.db = WordDatabase(env.DB_PATH)
    
    def tearDown(self):
        self.db.conn.close()
    
    def test_vocab_size(self):
        vocab_path = env.VOCAB_EMB_PATH

        vocab_emb = np.load(vocab_path)
        vocab_size = vocab_emb.shape[0]

        pruned_vocab_words = self.db.get_pruned_vocab()
        pruned_size = len(pruned_vocab_words)

        self.assertEqual(vocab_size, pruned_size)

if __name__ == "__main__":
    unittest.main()