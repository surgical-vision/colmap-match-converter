import h5py
import numpy as np
import os
from collections import defaultdict
import torch
from copy import deepcopy

# Define ANSI escape sequences for colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

def convert_matches_h5(match_dict, output_path, matcher, verbose=False, img_dir=None):
    """
    Convert matches to colmap format
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if not os.path.exists(f'{output_path}/matches_{matcher}.h5'):
        mode = 'w'
    else:
        mode = 'a'

    with h5py.File(f'{output_path}/matches_{matcher}.h5', mode) as f_match:
        key1, key2, = match_dict['img0_fn'], match_dict['img1_fn']
        # chiara saved all images as jpg so need to make sure if png replace fname with jpg
        if img_dir is not None:
            path_1 = os.path.join(img_dir, key1)
            path_2 = os.path.join(img_dir, key2)
            if not os.path.isfile(path_1):
                if key1.endswith('.jpg'):
                    key1 = key1.replace('.jpg', '.png')
                elif key1.endswith('.png'):
                    key1 = key1.replace('.png', '.jpg')
            if not os.path.isfile(path_2):
                if key2.endswith('.jpg'):
                    key2 = key2.replace('.jpg', '.png')
                elif key2.endswith('.png'):
                    key2 = key2.replace('.png', '.jpg')
        
        mkpts0, mkpts1 = match_dict['mkpts0'], match_dict['mkpts1']

        # check that mkpts0 and mkpts1 are both not empty
        if mkpts0.size == 0 or mkpts1.size == 0:
            # print(f'{RED}No matches found between {key1} and {key2}{RESET}')
            return

        if verbose:
            print(f'==============================================================================')
            print(f'{CYAN}Adding {len(mkpts0)} matches between {key1} and {key2}{RESET}')

        group1 = f_match.require_group(key1)

        if key2 in group1:
            print(f'{YELLOW}Key {key2} already exists in group {key1}{RESET}')
            existing_matches = group1[key2][...]
            new_matches = np.concatenate([mkpts0, mkpts1], axis=1)
            all_matches = np.vstack([existing_matches, new_matches])
            del group1[key2]
            group1.create_dataset(key2, data=all_matches)
        else:
            group1.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))
        
    return

def convert_matches_colmap(output_path, matcher):
    """
    Convert matches to colmap format
    """
    # check if h5 file exists 
    if not os.path.exists(f'{output_path}/matches_{matcher}.h5'):
        print(f'{RED}No matches_{matcher}.h5 file found in {output_path}{RESET}')
        return

    # Initialize dictionaries to store keypoints, matches, and total keypoints count.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)

    # Load existing match data from HDF5 file.
    with h5py.File(f'{output_path}/matches_{matcher}.h5', mode='r') as f_match:
        # Iterate over each group of matches identified by image keys.
        for k1 in f_match.keys():
            group  = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                # Accumulate keypoints from the matches for both images.
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                # Create match indices adjusted for the offset of accumulated keypoints.
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0]+=total_kpts[k1]
                current_match[:, 1]+=total_kpts[k2]
                total_kpts[k1]+=len(matches)
                total_kpts[k2]+=len(matches)
                match_indexes[k1][k2]=current_match

    # Consolidate and round keypoints for all images.
    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    
    # Create unique keypoints and matches for each image.
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
  
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            if m2.size == 0 or k1 not in unique_match_idxs or k2 not in unique_match_idxs:
                continue
            if unique_match_idxs[k1].size(0) == 0 or unique_match_idxs[k2].size(0) == 0:
                continue
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            if unique_kpts[k1].shape[0] == 0 or unique_kpts[k2].shape[0] == 0:
                continue
            if unique_kpts[k1].shape[1] != 2 or unique_kpts[k2].shape[1] != 2:
                print(f"{RED}Error: keypoints for {k1} or {k2} do not have 2 columns{RESET}")
                continue
            if m2.shape[1] != 2:
                print(f"{RED}Error: matches m2 do not have 2 columns: {m2.shape}{RESET}")
                continue
            
            # Fix starts here
            if unique_kpts[k1][m2[:, 0]].ndim == 1:
                mkpts_k1 = unique_kpts[k1][m2[:, 0]].reshape(-1, 2)
            else:
                mkpts_k1 = unique_kpts[k1][m2[:, 0]]

            if unique_kpts[k2][m2[:, 1]].ndim == 1:
                mkpts_k2 = unique_kpts[k2][m2[:, 1]].reshape(-1, 2)
            else:
                mkpts_k2 = unique_kpts[k2][m2[:, 1]]

            mkpts = np.concatenate([mkpts_k1, mkpts_k2], axis=1)
            # Fix ends here
            
            if mkpts.ndim != 2 or mkpts.shape[1] != 4:
                print(f"{RED}Error: mkpts has incorrect dimensions: {mkpts.shape}{RESET}")
                continue
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()

    
    # Save unique keypoints to HDF5 file.
    with h5py.File(f'{output_path}/keypoints.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1
    
    # Save unique matches to HDF5 file.
    with h5py.File(f'{output_path}/matches.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group  = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
            
            print(f"{GREEN}Saved COLMAP matches for {k1} in {output_path}/matches.h5{RESET}")
    
    return

def get_unique_idxs(A, dim=0):
    if A.size(0) == 0:
        return torch.tensor([], dtype=torch.long, device=A.device)

    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    
    if counts.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=A.device)

    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices