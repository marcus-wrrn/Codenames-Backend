from flask import Flask, request, jsonify
from flask_cors import CORS
from src.database.orm import Database
from src.utils.utilities import map_team, console_logger
from src.utils.model_loader import ModelLoader
import torch
import env
import random
from src.utils.word_board import init_gameboard
import logging


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model, vocab = init_model_and_vocab(MODEL_PATH, dataset, device=device)
logger = console_logger('console_logger')
logger.info('Initializing ModelLoader')
loader = ModelLoader(env.MODEL_PATH, env.DB_PATH, env.VOCAB_EMB_PATH, env.BOARD_EMB_PATH)

logger.info('Initializing Server')
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
    if first_team == bot_team: hint_word = loader.get_hint_word(board, bot_team)
    if first_team == human_team: hint_word = loader.get_hint_word(board, human_team)
        

    data = {
        'words': board.to_dict(),
        'hint_word': hint_word
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)