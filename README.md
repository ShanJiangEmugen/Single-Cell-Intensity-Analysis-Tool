# Cell Intensity Analysis Tool
A lightweight pipeline for cell segmentation and fluorescence intensity extraction using **CellPose** + **AICSImageIO**.    
Supports batch processing, per-cell intensity quantification, multi-channel analysis, and group-based statistics.    

## Features
- CellPose-based cell segmentation
- Extract per-cell intensity (median/mean) from any channel
- Batch processing of `.czi` images
- Automatic folder + CSV + figure output
- Optional multi-channel comparison
- Optional cell counting per channel
- Jupyter Notebook example included   

## Installation

Create environment:    
```
conda create -n cellpose python=3.10 
conda activate cellpose
```

Install dependencies:
```
pip install -r requirements.txt
```

## Configuration
Edit parameters in config.yaml:
```
diameter: 75
flow_threshold: 0.25
cellprob_threshold: 5
min_size: 2000
```

## Quick Start
### 1. Initialize project
```
from src.image_class import project
import yaml

with open("config.yaml") as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

group_info = {
    "Group_A": [1,5,6],
    "Group_B": [4,9,16]
}

p = project(
    folder="your_image_folder",
    param=param,
    cell_size_threshold=2000,
    group_info=group_info,
)
```

### 2. Run CellPose + intensity extraction
```
median_df, mean_df = p.batch_masking(
    cellpose_ch=0,
    signal_ch=1,
    gamma=1
)
```
Results will be saved automatically in a new timestamped folder.     

## Optional Tools
### Organize group data
```
p.orgnize_master_files(median_df, "median")
p.orgnize_master_files(mean_df, "mean")
```
### Two-channel comparison
```
p.both_channel_ratio(
    cellpose_ch=0,
    mask_ch1=1, mask_ch2=2,
    low_bound_ch1=100, up_bound_ch1=800,
    low_bound_ch2=50,  up_bound_ch2=600,
)
```
### Count cells per channel
```
raw, sorted = p.count_cells_per_channel([0,1,2])
```

## Notebook
A ready-to-run demo is included in:
`Colab_Version/Cell_intensity_tool.ipynb`

## License
MIT License




