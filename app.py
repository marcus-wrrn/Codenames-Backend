from flask import Flask, request, jsonify
from flask_cors import CORS
from src.utils.utilities import map_team, console_logger, get_random_team
from src.utils.model_loader import ModelLoader
import env
from src.views.word_board import init_gameboard, create_board_from_response
from src.views.gameturn import GameLog
from src.database.gamelog import GameLogDatabase


logger = console_logger('console_logger')
logger.info('Initializing ModelLoader')
loader = ModelLoader()

logger.info('Initializing Server')
app = Flask(__name__)

CORS(app) 

@app.route('/api/startgame')
def start_game():
    first_team = get_random_team()

    board = init_gameboard(env.DB_PATH)

    # Play turn
    turn_data = loader.play_turn_algorithmic(board, first_team)
    if not turn_data:
        return jsonify({'error': 'Error generating hint word'}), 500
    
    hint_word, sim_word_keys, sim_scores = turn_data

    data = {
        'words': board.to_dict(),
        'hint_info': {
            'hint_word': hint_word,
            'sim_word_ids': sim_word_keys,
            'sim_scores': sim_scores
        },
        'current_team': first_team.value
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
    turn_data = loader.play_turn_algorithmic(board, team)
    if not turn_data:
        return jsonify({'error': 'Error generating hint word'}), 500
    
    hint_word, sim_word_keys, sim_scores = turn_data

    data = {
        'words': board.to_dict(),
        'hint_info': {
            'hint_word': hint_word,
            'sim_word_ids': sim_word_keys,
            'sim_scores': sim_scores
        },
        'current_team': team.value
    }

    return jsonify(data)

@app.route('/api/savelog', methods=['POST'])
def save_log():
    data = request.json
    if 'save_info' not in data:
        return jsonify({'error': 'Missing required fields: save_info'}), 400
    
    game_info = data['save_info']
    if 'log' not in game_info and 'game_words' not in game_info and 'word_colorIDs' not in game_info:
        return jsonify({'error': 'Missing required fields'}), 400
    try:
        log = GameLog(game_info['log'], env.DB_PATH)
        game_words = game_info['game_words']
        word_colorIDs = game_info['word_colorIDs']
        origin_ip = request.remote_addr
    except Exception as e:
        return jsonify({'error processing data': str(e)}), 500
    
    try:
        with GameLogDatabase(collection_name="logs_with_ids") as db:
            db.save_log(log, game_words, word_colorIDs, origin_ip)

    except Exception as e:
        return jsonify({'error saving log': str(e)}), 500
    
    return jsonify({'success': True, 'log': log.to_dict() })

@app.route('/api/get_sim_texts', methods=['POST'])
def get_sim_texts():
    data = request.json
    if 'word' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    word = data['word']
    num_results = 20
    if 'num_results' in data:
        num_results = data['num_results'] if type(data['num_results']) == int else num_results
    try:
        texts, scores, avg_score = loader.search_vocabulary(word, num_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    data = {
        'texts': texts,
        'scores': scores,
        'avg_score': avg_score
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)