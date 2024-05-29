from torch.utils.data import Dataset
import torch
import json

class CodeGiverDataset(Dataset):
    def __init__(self, vocab_dir: str, board_dir: str):
        super().__init__()
        
        self.vocab_dir = vocab_dir
        self.board_dir = board_dir

        self.text_data = self._load_data(self.vocab_dir)
        self.code_dict = self._create_board_dict(self.text_data)
        self.vocab_dict = self._create_guess_dict(self.text_data)
        
        with open(self.board_dir, 'r') as fp:
            self.game_data = json.load(fp)
        
        self._process_game_data(self.game_data) 

    def _load_data(self, codename_dir: str):
        with open(codename_dir, 'r') as fp:
            filedata = json.load(fp)
        return filedata       
    
    # Intializers
    def _create_board_dict(self, data: json):
        return { text: embedding for text, embedding in zip(data['codewords'], data['code_embeddings']) }
    
    def _create_guess_dict(self, data: json):
        return { text: embedding for text, embedding in zip(data['guesses'], data['guess_embeddings']) }
    
    def _process_game_data(self, data: json):
        self.positive_sents = data['positive']
        self.negative_sents = data['negative']
        self.neutral_sents = data['neutral']

    # Accessors
    def get_vocab(self, guess_data=True):
        words = []
        embeddings = []
        data = self.vocab_dict.items() if guess_data else self.code_dict.items()
        for key, value in data:
            words.append(key)
            embeddings.append(value)
        return words, embeddings
    
    def get_pruned_vocab(self):
        """Returns all words/embeddings in the guess dataset that are not in the codename dataset"""
        words = []
        embeddings = []
        guess_data = self.vocab_dict.items()
        for key, value in guess_data:
            # If word exists in the code words set remove it
            if key in self.code_dict:
                continue
            words.append(key)
            embeddings.append(value)
        return words, embeddings
    
    def __len__(self):
        return len(self.positive_sents)
    
    def __getitem__(self, index):
        pos_sent = self.positive_sents[index]
        neg_sent = self.negative_sents[index]

        # Get embeddings
        pos_embeddings = torch.stack([torch.tensor(self.code_dict[word]) for word in pos_sent.split(' ')])
        neg_embeddings = torch.stack([torch.tensor(self.code_dict[word]) for word in neg_sent.split(' ')])

        return pos_sent, neg_sent, pos_embeddings, neg_embeddings

class CodeNamesDataset(CodeGiverDataset):
    """
    Dataset for the full game of Codenames, containing both positive, negative, neutral and the assasin words
    """
    def __init__(self, code_dir: str, game_dir: str, seperator=' '):
        super().__init__(code_dir, game_dir)
        self.sep = seperator
        self.assassin_words = self.game_data['assassin']

    def __getitem__(self, index):
        pos_sent = self.positive_sents[index]
        neg_sent = self.negative_sents[index]
        neutral_sent = self.neutral_sents[index]
        assassin_word = self.assassin_words[index]

        # Get embeddings
        pos_embeddings = torch.stack([torch.tensor(self.code_dict[word]) for word in pos_sent.split(self.sep)])
        neg_embeddings = torch.stack([torch.tensor(self.code_dict[word]) for word in neg_sent.split(self.sep)])
        neutral_embeddings = torch.stack([torch.tensor(self.code_dict[word]) for word in neutral_sent.split(self.sep)])
        assassin_embedding = torch.tensor(self.code_dict[assassin_word])

        return (pos_sent, neg_sent, neutral_sent, assassin_word), (pos_embeddings, neg_embeddings, neutral_embeddings, assassin_embedding)