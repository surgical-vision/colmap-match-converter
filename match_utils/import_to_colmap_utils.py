from .colmap_database_utils import COLMAPDatabase, image_ids_to_pair_id
import h5py
import os
import warnings
import numpy as np
from PIL import Image, ExifTags
from tqdm import tqdm

RED = '\033[91m' # bugs
GREEN = '\033[92m' # success
RESET = '\033[0m'
YELLOW = '\033[93m' # warnings

def import_into_colmap(img_dir,
                       output_path,
                       database_path = 'colmap.db', verbose=False):
    if verbose:
        print(f'Attempting to connect to database:\n       {database_path}')
    db = COLMAPDatabase.connect(database_path)
    if verbose:
        print(f'{GREEN}Connected to database:\n       {database_path}{RESET}')
    db.create_tables()
    if verbose:
        print(f'Created tables')
    single_camera = False
    fname_to_id = add_keypoints(db, output_path, img_dir, 'simple-radial', single_camera)
    add_matches(db, output_path, fname_to_id)
    db.commit()
    return

def add_keypoints(db, h5_path, image_path, camera_model, single_camera = False):
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

    camera_id = None
    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys()), colour='magenta', desc='Adding keypoints'):
            
        keypoints = keypoint_f[filename][()]

        img_ext = os.path.splitext(filename)[1]

        path = os.path.join(image_path, filename)
        if not os.path.isfile(path):
            # if not os.path.isfile replace jpg with png and try again
            if img_ext == '.jpg':
                filename = filename.replace('.jpg', '.png')
                path = os.path.join(image_path, filename)
            else:
                filename = filename.replace('.png', '.jpg')
                path = os.path.join(image_path, filename)
            if not os.path.isfile(path):
                raise IOError(f'{RED}Image {path} not found{RESET}')

        if camera_id is None or not single_camera:
            camera_id = create_camera(db, path, camera_model)
        image_id = db.add_image(filename, camera_id)
        fname_to_id[filename] = image_id

        db.add_keypoints(image_id, keypoints)

    return fname_to_id

def create_camera(db, image_path, camera_model):
    image         = Image.open(image_path)
    width, height = image.size

    focal = get_focal(image_path)

    if camera_model == 'simple-pinhole':
        model = 0 # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == 'pinhole':
        model = 1 # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == 'simple-radial':
        model = 2 # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == 'opencv':
        model = 4 # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
         
    return db.add_camera(model, width, height, param_arr)

def get_focal(image_path, err_on_default=False):

    image         = Image.open(image_path)
    max_size      = max(image.size)
    
    # Retrieve EXIF data from the image, which contains meta data 
    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # Iterate over the EXIF data to find the 'FocalLengthIn35mmFilm' tag.
        # This tag gives the focal length scaled to a 35mm film camera equivalent.
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break
        # If a 35mm equivalent focal length is found, calculate the actual focal length using the image's max size.
        if focal_35mm is not None:
            focal = focal_35mm / 35. * max_size
    
    # If the focal length is not found in the EXIF data, use a default value.
    if focal is None:
        if err_on_default:
            raise RuntimeError("{RED}Failed to find focal length{RESET}")

        # default focal length using a predefined prior (1.2).
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal

def add_matches(db, h5_path, fname_to_id):
    match_file = h5py.File(os.path.join(h5_path, 'matches.h5'), 'r')
    
    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_1 = fname_to_id[key_1]
                id_2 = fname_to_id[key_2]

                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f'{YELLOW}Pair {pair_id} ({id_1}, {id_2}) already added!{RESET}')
                    continue
            
                matches = group[key_2][()]
                db.add_matches(id_1, id_2, matches)

                added.add(pair_id)

                pbar.update(1)