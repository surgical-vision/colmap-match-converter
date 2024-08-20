import torch
import argparse
import os
from pathlib import Path

from match_utils.read_matches_utils import read_matches
from match_utils.convert_matches_utils import convert_matches_h5, convert_matches_colmap 
from match_utils.import_to_colmap_utils import import_into_colmap

# Define ANSI escape sequences for colors
CYAN = '\033[0m'
RESET = '\033[0m'

def read_all_matches(matches_path, verbose):
    # if length of matches_path.glob('*') is too big then it will run out of input
    matches_list = []
    matches_path = Path(matches_path)
    if matches_path.is_dir():
        for match_file in matches_path.glob('*'):
            matches_list.append(read_matches(match_file, verbose))
    else:
        matches_list.append(read_matches(matches_path, verbose))
    return matches_list

def read_and_convert_matches(match_files, output_path, matcher, verbose, img_dir):
    for match_file in match_files:
        match_data = read_matches(match_file, verbose)
        # if match_data is not None then convert otherwise continue
        if match_data is not None:
            convert_matches_h5(match_data, output_path, matcher, verbose, img_dir=img_dir)
        else:
            continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert matches to colmap db')
    parser.add_argument('--matches', type=str, help='path to matches')
    parser.add_argument('--img_dir', type=str, help='path to images')
    parser.add_argument('--output_path', type=str, help='path to output')
    parser.add_argument('--matcher', type=str, help='matcher used')
    parser.add_argument('--verbose', action='store_true', default=False, help='print more information')
    args = parser.parse_args()

    # clear output path 
    if os.path.exists(args.output_path):
        os.system(f'rm -r {args.output_path}')
        os.makedirs(args.output_path)
    else:
        os.makedirs(args.output_path)

    # -----------------------------------------
    # read matches
    # -----------------------------------------

    print(f'{CYAN}=============================================================================={RESET}')
    print(f'{CYAN}Reading matches{RESET}')
    # must do this in steps if matches are too big
    # check size of matches_path.glob('*') and if too big then do in steps
    matches_path = Path(args.matches)
    if matches_path.is_dir():
        all_files = list(matches_path.glob('*'))
        batch_size = 1000
        for start_idx in range(0, len(all_files), batch_size):
            end_idx = start_idx + batch_size
            batch_files = all_files[start_idx:end_idx]
            print(f'{CYAN}Processing batch from {start_idx} to {end_idx - 1}{RESET}')
            read_and_convert_matches(batch_files, args.output_path, args.matcher, args.verbose, args.img_dir)

    # -----------------------------------------
    # convert matches
    # -----------------------------------------

    print(f'{CYAN}=============================================================================={RESET}')
    # for match in example:
        # convert_matches_h5(match, args.output_path, args.matcher, args.verbose, img_dir=args.img_dir)
    convert_matches_colmap(args.output_path, args.matcher)

    # -----------------------------------------
    # import to colmap db 
    # -----------------------------------------

    print(f'{CYAN}=============================================================================={RESET}')
    database_path = args.output_path + '/colmap.db'
    import_into_colmap(args.img_dir, args.output_path, database_path, args.verbose)