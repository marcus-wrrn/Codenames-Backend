from src.database.orm import Database
from src.database.vector_search import VectorSearch
from env import VOCAB_DIR, DB_PATH, BOARD_EMB_PATH, VOCAB_EMB_PATH
import numpy as np
import json

"""
script used to fill a database with the words from a json file used in training the model
allows for faster access to game data

word data is stored in an sqlite database, embeddings are stored in a npy file where word_id is assosciated with values in the db
"""

print('Opening file')
with open(VOCAB_DIR, 'r') as f:
    word_data = json.load(f)

# Extract data
board_words = word_data['codewords']


vocab_words = word_data['guesses']


# Fill database
with Database(DB_PATH) as db:
    print('Loading board data')
    for i, word in enumerate(board_words):
        db.insert_board(i, word, commit=False)
    db.conn.commit()
    print('Loading vocab data')
    for i, word in enumerate(vocab_words):
        db.insert_vocab(i, word, commit=False)
    db.conn.commit()
    # Retrieve pruned vocab words to make sure we aren't saving unnessecary data to the vocab embedding file
    vocab_words = db.get_pruned_vocab()

print('Processing embeddings')
# Process board embeddings
board_embeddings = word_data['code_embeddings']
board_embeddings = np.array(board_embeddings, dtype=np.float32)
np.save(BOARD_EMB_PATH, board_embeddings)

# Process vocab embeddings
vocab_embeddings = word_data['guess_embeddings']

# Prune duplicate embeddings used in the board
vocab_embeddings = [vocab_embeddings[word[1]] for word in vocab_words]

vocab_embeddings = np.array(vocab_embeddings, dtype=np.float32)
np.save(VOCAB_EMB_PATH, vocab_embeddings)

