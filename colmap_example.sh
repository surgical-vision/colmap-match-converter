python match_converter.py --matches "$matches_path" --output_path "$out_directory" --img_dir "$img_dir" --matcher "aliked-lg"

python colmap_reconstructer.py --input_path "$out_directory" --image_path "$img_dir"