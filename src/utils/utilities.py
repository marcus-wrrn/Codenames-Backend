import torch
from torch import Tensor
import torch.nn.functional as F
import random
import logging
from src.views.word_board import WordColor

def get_device(is_cuda: str):
    if (is_cuda.lower() == 'y' and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")

def convert_args_str_to_bool(arg: str):
    return True if arg.lower() == 'y' else False

def calc_game_scores_no_assasin(model_out: Tensor, pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor, device: torch.device):
    model_out_expanded = model_out.unsqueeze(1)
    pos_scores = F.cosine_similarity(model_out_expanded, pos_encs, dim=2)
    neg_scores = F.cosine_similarity(model_out_expanded, neg_encs, dim=2)
    neut_scores = F.cosine_similarity(model_out_expanded, neut_encs, dim=2)

    combined_scores = torch.cat((pos_scores, neg_scores, neut_scores), dim=1)
    _, indices = combined_scores.sort(dim=1, descending=True)

    # create reward copies
    pos_reward = torch.zeros(pos_scores.shape[1]).to(device)
    neg_reward = torch.ones(neg_scores.shape[1]).to(device) * 2
    neut_reward = torch.ones(neut_scores.shape[1]).to(device) 

    combined_rewards = torch.cat((pos_reward, neg_reward, neut_reward))
    # Make shape [batch_size, total_number_of_embeddings]
    combined_rewards = combined_rewards.expand((combined_scores.shape[0], combined_rewards.shape[0]))
    # Retrieve the ordered number of rewards, in the order of highest cossine similarity
    rewards = torch.gather(combined_rewards, 1, indices)
    # set all target embeddings to 0 and unwanted embeddings to 1
    non_zero_mask = torch.where(rewards != 0, 1., 0.)
    # Find the total number of correct guesses, equal to the index of the first non-zero value in the mask
    num_correct = torch.argmax(non_zero_mask, dim=1)
    # Find the first incorrect value
    first_incorrect_value = rewards[torch.arange(rewards.size(0)), num_correct]
    return num_correct.float().mean(), first_incorrect_value.mean() - 1

def calc_codenames_score(model_out: Tensor, pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor, assas_encs: Tensor, device: torch.device):
    model_out_expanded = model_out.unsqueeze(1)
    assas_expanded = assas_encs.unsqueeze(1)

    pos_scores = F.cosine_similarity(model_out_expanded, pos_encs, dim=2)
    neg_scores = F.cosine_similarity(model_out_expanded, neg_encs, dim=2)
    neut_scores = F.cosine_similarity(model_out_expanded, neut_encs, dim=2)
    assas_scores = F.cosine_similarity(model_out_expanded, assas_expanded, dim=2)

    combined_scores = torch.cat((pos_scores, neg_scores, neut_scores, assas_scores), dim=1)
    _, indices = combined_scores.sort(dim=1, descending=True)

    # create reward copies
    pos_reward = torch.zeros(pos_scores.shape[1]).to(device)
    neg_reward = torch.ones(neg_scores.shape[1]).to(device) * 2
    neut_reward = torch.ones(neut_scores.shape[1]).to(device) 
    assas_reward = torch.ones(assas_scores.shape[1]).to(device) * 3

    combined_rewards = torch.cat((pos_reward, neg_reward, neut_reward, assas_reward))
    # Make shape [batch_size, total_number_of_embeddings]
    combined_rewards = combined_rewards.expand((combined_scores.shape[0], combined_rewards.shape[0]))
    # Retrieve the ordered number of rewards, in the order of highest cosine similarity
    rewards = torch.gather(combined_rewards, 1, indices)
    # set all target embeddings to 0 and unwanted embeddings to 1
    non_zero_mask = torch.where(rewards != 0, 1., 0.)
    # Find the total number of correct guesses, equal to the index of the first non-zero value in the mask
    num_correct = torch.argmax(non_zero_mask, dim=1)
    # Find the first incorrect value
    first_incorrect_value = rewards[torch.arange(rewards.size(0)), num_correct]

    assassin_sum = torch.sum(first_incorrect_value == 3, dim=0)
    neg_sum = torch.sum(first_incorrect_value == 2, dim=0)
    neut_sum = torch.sum(first_incorrect_value == 1, dim=0)

    return num_correct.float().mean(), neg_sum, neut_sum, assassin_sum\

def slice_board_embeddings(embs: Tensor):
    """Used to randomly remove words/collections of text, from the game board"""
    rand_num = random.randint(1, embs.shape[1])
    return embs[:, :rand_num, :]


def cluster_embeddings(embs: Tensor, dim=1):
    """Mean pool and normalize all embeddings"""
    out = torch.mean(embs, dim=dim)
    out = F.normalize(out, p=2, dim=dim)
    return out

def console_logger(logger_name: str, level=logging.DEBUG) -> logging.Logger:
    """Initializes a logger for the use of printing debug messages to the console"""
    console_logger = logging.getLogger(logger_name)
    console_logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    console_logger.addHandler(stream_handler)

    return console_logger


def map_team(team: int | None):
    if team is None:
        return None
    
    if team == 1:
        return WordColor.RED
    return WordColor.BLUE

def get_random_team():
    return random.choice([WordColor.RED, WordColor.BLUE])