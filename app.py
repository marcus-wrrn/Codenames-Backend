from flask import Flask, request, jsonify
from flask_cors import CORS
from src.setup import init_model_and_vocab
from src.dataset import CodeNamesDataset
from src.database.orm import Database
import torch
import env
import random
from src.utils.data_objs import WordColor, Word


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model, vocab = init_model_and_vocab(MODEL_PATH, dataset, device=device)

app = Flask(__name__)

CORS(app) 

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/api/startgame')
def start_game():
    with Database(env.DB_PATH) as db:
        board = db.get_board()
    
    all_words = []
    for i, word_obj in enumerate(board):
        word = word_obj[0]

        color = WordColor.GREY
        if i < 9:
            color = WordColor.RED
        elif i < 18:
            color = WordColor.BLUE
        elif i == len(board) - 1:
            color = WordColor.BLACK
        
        all_words.append(Word(i, word, color))
    random.shuffle(all_words)

    data = {
        'words': [word.to_dict() for word in all_words],
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)