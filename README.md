# colmap-match-converter

Converts image matches to COLMAP format.

## Important to Note:

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

However, keep in mind their repository outputs matches in relation to rescaled images (i.e., 512x512). Some minor adjustments need to be made to the `basematcher` class to output the un-rescaled (or re-rescaled, if you will) matches in relation to the original images. We made these minor changes in our repo, which is built on their repo and can be found here: [link].

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
**> **Note:** You can also use these matches with [glomap](https://github.com/colmap/glomap) by creating a script like `colmap_reconstructer.py` for glomap.**

## Citation:

If you can make use of this in your own research, please be so kind as to cite our paper:

```bibtex
@Article{,
  author       = {},
  title        = {},
  journal      = {},
  number       = {},
  volume       = {},
  month        = {},
  year         = {},
  url          = {}
}
```
