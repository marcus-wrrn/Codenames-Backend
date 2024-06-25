from src.utils.model_loader import ModelLoader
from src.database.orm import WordDatabase
from env import DB_PATH
import numpy as np

animal_filepath = "./data/animals.txt"
animal_embedding_path = "./data/animal_embeddings.npy"

with open(animal_filepath, 'r') as f:
    animals = f.read().splitlines()
# process animals
animals = [animal.lower().strip() for animal in animals]
animals.sort()

loader = ModelLoader()

with WordDatabase(DB_PATH) as db:
    for i, animal in enumerate(animals):
        db.insert_animal_board(i, animal, commit=False)
    db.conn.commit()
    print('Animal data loaded')

# encode animals
animal_embeddings = loader.semantic_backbone.encode(animals, convert_to_numpy=True)
np.save(animal_embedding_path, animal_embeddings)
