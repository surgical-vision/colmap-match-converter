import argparse
import pycolmap
from time import time
import os
import sys
from copy import deepcopy
import numpy as np
from read_write_colmap import read_cameras_binary
from contextlib import contextmanager

# Define ANSI escape sequences for colors
RED = '\033[91m' # bugs
GREEN = '\033[92m' # success
CYAN = '\033[0m'
RESET = '\033[0m'

@contextmanager
def suppress_output():
    """Suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        # Save the current stdout and stderr file descriptors.
        old_stdout_fileno = os.dup(sys.stdout.fileno())
        old_stderr_fileno = os.dup(sys.stderr.fileno())

        try:
            # Redirect stdout and stderr to /dev/null.
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())

            yield
        finally:
            # Restore the original stdout and stderr file descriptors.
            os.dup2(old_stdout_fileno, sys.stdout.fileno())
            os.dup2(old_stderr_fileno, sys.stderr.fileno())

def run_colmap(database_path, args):
    print(f'{CYAN}=============================================================================={RESET}')
    t=time()
    pycolmap.match_exhaustive(database_path)
    t=time()-t
    print(f'{CYAN}=============================================================================={RESET}')
    print(f'{GREEN}Geometric Verification in {t:.2f} seconds{RESET}')
    print(f'{CYAN}=============================================================================={RESET}')

    t=time()
    mapper_options=pycolmap.IncrementalPipelineOptions({'min_model_size': 3, 'min_num_matches': 5, 'max_num_models': 2})
    with suppress_output():
        maps = pycolmap.incremental_mapping(database_path, image_path=args.image_path, output_path=args.output_path, options=mapper_options)
    t=time()-t
    print(f'{CYAN}=============================================================================={RESET}')
    print(f'{GREEN}Reconstruction done in {t:.2f} seconds{RESET}')
    print(f'{CYAN}=============================================================================={RESET}')

    imgs_registered=0
    best_idx=None
    if isinstance(maps, dict):
        for idx1, rec in maps.items():
            if len(rec.images) > imgs_registered:
                imgs_registered = len(rec.images)
                best_idx = idx1

    return best_idx, maps

def load_all_cameras(output_path, num_reconstructions):
    all_cameras = {}
    for i in range(num_reconstructions):
        camera_path = os.path.join(output_path, str(i), 'cameras.bin')
        if os.path.exists(camera_path):
            cameras = read_cameras_binary(camera_path)
            all_cameras.update(cameras)
        else:
            print(f'{RED}No cameras.bin found at {camera_path}{RESET}')
    return all_cameras

if __name__ == '__main__':

    example_image_path = '/home/chiara/workspace/image-matching/eccv-paper-2024/image-matching-models/output/imm/scene_0/scene_processed_images'

    parser = argparse.ArgumentParser(description='run ransac and reconstruct')
    parser.add_argument('--input_path', type=str, default='output/superpoint', help='path to directory containing colmap db')
    parser.add_argument('--output_path', type=str, default=None, help='path to output directory. default is input_path')
    parser.add_argument('--image_path', type=str, default=example_image_path, help='path to images, should be the scene_processed_images')
    parser.add_argument('--verbose', action='store_true', default=False, help='print more information')
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = args.input_path
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"{RED}Input path {args.input_path} not found{RESET}")
    
    # find colmap db in input path
    if "colmap.db" in os.listdir(args.input_path):
        database_path = os.path.join(args.input_path, "colmap.db")
    else:
        raise FileNotFoundError(f"{RED}No colmap.db found in {args.input_path}{RESET}")
    
    best_idx, maps = run_colmap(database_path, args)

    cameras = load_all_cameras(args.output_path, len(maps))

    # need to create a poses.txt with f.write('image_name,rotation_matrix,translation_vector,calibration_matrix,dataset,scene\n')
    if best_idx is not None: 
        if args.verbose:
            print(maps[best_idx].summary())
            print(f'{CYAN}=============================================================================={RESET}')
        
        scene = args.image_path.split('/')[-2]

        with open(os.path.join(args.output_path, 'predicted_poses.csv'), 'w') as f:
            for k, im in maps[best_idx].images.items():
                img_name = im.name
                rotation_dir = deepcopy(im.cam_from_world.rotation.matrix())
                translation_vec = deepcopy(np.array(im.cam_from_world.translation))
                camera_id = im.camera_id

                # Debug output to verify camera IDs
                if args.verbose:
                    print(f'{CYAN}Image: {img_name}, Camera ID: {camera_id}{RESET}')
                    print(f'{CYAN}Available Camera IDs: {list(cameras.keys())}{RESET}')

                if cameras and camera_id in cameras:
                    camera = cameras[camera_id]
                    calibration_matrix = np.array([
                        [camera.params[0], 0, camera.params[1]],
                        [0, camera.params[0], camera.params[2]],
                        [0, 0, 1]
                    ])
                else:
                    print(f'{RED}Camera ID {camera_id} not found in cameras{RESET}')
                    calibration_matrix = np.eye(3)
                
                rotation_dir_str = ';'.join(map(str, rotation_dir.flatten()))
                translation_vec_str = ';'.join(map(str, translation_vec))
                calibration_matrix_str = ';'.join(map(str, calibration_matrix.flatten()))

                dataset = 'imc24' if 'imc' in args.image_path or 'imm' in args.image_path else 'niantic'
                f.write(f'{img_name},{rotation_dir_str},{translation_vec_str},{calibration_matrix_str},{dataset},{scene}\n')
    else:
        print(f'{RED}No reconstruction found{RESET}')
        print(f'{CYAN}=============================================================================={RESET}')
        print(f'{RED}Exiting{RESET}')
        exit(0)
    
    if args.verbose:
        print(f'{GREEN}Predicted poses saved at {args.output_path}/predicted_poses.csv{RESET}')
        print(f'{CYAN}=============================================================================={RESET}')
    
    # print total number of registered images vs total number of images
    total_images = len(os.listdir(args.image_path))
    registered_images = len(maps[best_idx].images)
    print(f'{GREEN}Registered {registered_images}/{total_images} images{RESET}')
    print(f'{CYAN}=============================================================================={RESET}')
    with open(os.path.join(args.output_path, 'registered_images.txt'), 'w') as f:
        f.write(f'Registered images for scene {scene} for reconstruction {best_idx}\n')
        f.write(f'{registered_images}/{total_images}\n')
    print(f'{GREEN}Registered images saved at {args.output_path}/registered_images.txt{RESET}')