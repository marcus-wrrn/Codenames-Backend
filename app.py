from flask import Flask, request, jsonify
from flask_cors import CORS
from src.setup import init_model_and_vocab
from env import MODEL_PATH, VOCAB_DIR, BOARD_DIR

model, vocab = init_model_and_vocab(MODEL_PATH, VOCAB_DIR, BOARD_DIR)

app = Flask(__name__)

CORS(app) 

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()