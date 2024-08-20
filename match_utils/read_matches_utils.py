import torch 
import os 
from pathlib import Path

YELLOW = '\033[93m'
RESET = '\033[0m'

def read_matches(match_dict_path, verbose=False):
    """
    Read matches from a torch file
    """
    if match_dict_path.suffix != '.torch' and verbose:
        print(f'{YELLOW}File {match_dict_path} does not end in .torch{RESET}')
        return None
    try:
        match_dict = torch.load(match_dict_path)
    except:
        print(f'{YELLOW}Error reading {match_dict_path}{RESET}')
        return None
    match_dict['img0_fn'] = match_dict['img0_path'].name 
    match_dict['img1_fn'] = match_dict['img1_path'].name
    return match_dict