# colmap-match-converter

Converts image matches to COLMAP format.

## Note:

COLMAP requires matches to be in relation to the original input image scale/coordinate system. Within COLMAP's sparse and dense reconstruction processes, it will load images from the original path. Ensure that the matches you use are related to the original images and not rescaled images.

## Input Format:

This converter assumes you have Torch files with the following format:

```python
output_dict = {
    "mkpts0": mkpts0,  # array (N x 2), keypoints from img0 that match mkpts1 
    "mkpts1": mkpts1,  # array (N x 2), keypoints from img1 that match mkpts0 
    "img0_path": img0_path,  # path to img0  
    "img1_path": img1_path,  # path to img1
}
```
        
## Acknowledgements:

This repo was based on the matches created by, and we are very grateful for, the following repository:

- [Image Matching Models](https://github.com/gmberton/image-matching-models): This repository allows easy use of many image matching models and standardizes all their outputs. We recommend getting matches from this repository.

However, keep in mind their repository outputs matches in relation to rescaled images (i.e., 512x512). Some minor adjustments need to be made to the `basematcher` class to output the un-rescaled (or re-rescaled, if you will) matches in relation to the original images. We made these minor changes in our repo, which is built on their repo and will eventually be linked here: [link].

And of course, the COLMAP repository:

- [COLMAP](https://github.com/colmap/colmap)

## Using This Repo:

Install the following dependencies:
- `pycolmap`
- `numpy`
- `pytorch`
- `pillow`
- `tqdm`
- `h5py`

### Convert Matches:

```bash
python match_converter.py --matches "$model_dir" --output_path "$out_directory" --img_dir "$img_dir" --matcher "$matcher"
```
> **Note:** `matcher` is the name of the matcher method used, just for saving and naming purposes.

### Running COLMAP reconstruction
```bash
python colmap_reconstructer.py --input_path "$out_directory" --image_path "$img_dir"
```
If you want to adjust how this is run and modify the regular COLMAP reconstruction settings, `colmap_reconstructer.py` is where you would make those changes.

> **Note:** You can also use these matches with [glomap](https://github.com/colmap/glomap) by creating a script like `colmap_reconstructer.py` for glomap.

## Branches Overview:

### Main Branch: 
Intended to convert matches into a format compatible with COLMAP. The database can then be used with the standard COLMAP pipeline.

### Glomap Branch: 
The `glomap` branch introduces a modified approach to importing matches in a way that is compatible with [glomap](https://github.com/colmap/glomap). To use `glomap` with this custom match database, ensure you have already installed [glomap](https://github.com/colmap/glomap) along with its dependencies (`COLMAP`, `PoseLib`, etc.). Here's an example:

```bash
# Convert matches and generate a custom match database.
python match_converter.py --matches "$matches_path" --output_path "$out_directory" --img_dir "$img_dir" --matcher "aliked-lg"

# Use the resulting database in your glomap pipeline.
glomap mapper --database_path ./example/colmap.db --image_path $image_path --output_path ./example 
```

## Citation:

If you can make use of this in your own research, please be so kind as to cite our paper:

```bibtex
@article{bonilla2024mismatched,
  title={Mismatched: Evaluating the Limits of Image Matching Approaches and Benchmarks},
  author={Bonilla, Sierra and Di Vece, Chiara and Daher, Rema and Ju, Xinwei and Stoyanov, Danail and Vasconcelos, Francisco and Bano, Sophia},
  journal={arXiv preprint arXiv:2408.16445},
  year={2024}
}
```
