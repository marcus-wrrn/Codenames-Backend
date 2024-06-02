from flask import Flask, request, jsonify
from flask_cors import CORS
from src.utils.utilities import map_team, console_logger
from src.utils.model_loader import ModelLoader
import env
from src.utils.word_board import init_gameboard, create_board_from_response


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
    if 'first_team' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        first_team = map_team(data['first_team'])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    board = init_gameboard(env.DB_PATH)

    # Play turn
    turn_data = loader.play_turn(board, first_team)
    if not turn_data:
        return jsonify({'error': 'Error generating hint word'}), 500
    
    hint_word, sim_word_keys = turn_data

    data = {
        'board': board.to_dict(),
        'board_state': {
            'hint_word': hint_word,
            'sim_word_ids': sim_word_keys
        }
    }

    return jsonify(data)

@app.route('/api/playturn', methods=['POST'])
def play_turn():
    data = request.json
    if 'team' not in data or 'words' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        team = map_team(data['team'])
        board = create_board_from_response(env.DB_PATH, data['words'])
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    # Play turn
    turn_data = loader.play_turn(board, team)
    if not turn_data:
        return jsonify({'error': 'Error generating hint word'}), 500
    
    hint_word, sim_word_keys = turn_data

    data = {
        'board': board.to_dict(),
        'board_state': {
            'hint_word': hint_word,
            'sim_word_ids': sim_word_keys
        }
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)