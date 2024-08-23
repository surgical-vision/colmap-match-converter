# Author:   Sierra Bonilla
# Date:     23 08 24

#! /bin/bash 

matches_path='example/aliked-lg'
out_directory='example/output'
img_dir='example/images'

python match_converter.py --matches "$matches_path" --output_path "$out_directory" --img_dir "$img_dir" --matcher "aliked-lg"

python colmap_reconstructer.py --input_path "$out_directory" --image_path "$img_dir"