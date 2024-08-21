python match_converter.py --matches "/home/sierra/eccv-paper-2024/image-matching-models/output/niantic/s00460/seq1/aliked-lg" --output_path "example" --img_dir "/home/sierra/eccv-paper-2024/image-matching-models/data/niantic/val/s00460/seq1" --matcher "aliked-lg"

glomap mapper \
    --database_path ./example/colmap.db \
    --image_path    /home/sierra/eccv-paper-2024/image-matching-models/data/niantic/val/s00460/seq1 \
    --output_path   ./example