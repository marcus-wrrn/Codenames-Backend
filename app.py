from flask import Flask, request, jsonify
from flask_cors import CORS
from src.setup import init_model_and_vocab
from src.dataset import CodeNamesDataset
from src.database.orm import Database
from src.utils.utilities import map_team
import torch
import env
import random
from src.utils.data_objs import init_gameboard


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model, vocab = init_model_and_vocab(MODEL_PATH, dataset, device=device)

app = Flask(__name__)

CORS(app) 

@app.route('/')
def hello():
    return 'Hello!'

@app.route('/api/startgame', methods=['POST'])
def start_game():

    data = request.json
    if 'human_team' not in data or 'bot_team' not in data or 'first_team' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        human_team = map_team(data['human_team'])
        bot_team = map_team(data['bot_team'])
        first_team = map_team(data['first_team'])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    board = init_gameboard(env.DB_PATH)

    hint_word = ''
    if first_team == bot_team:
        with Database(env.DB_PATH) as db:
            db.cursor.execute('SELECT word FROM vocab WHERE word_id = 247')
            hint_word = db.cursor.fetchone()[0]

    data = {
        'words': board.to_dict(),
        'hint_word': hint_word
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)