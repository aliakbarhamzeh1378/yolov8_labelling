# YOLOv8 Object Labeling


This project provides a script for labeling objects in images using a YOLOv8-based pretrained model. It allows for efficient and accurate object detection and localization.

## Features
- Labels objects in images using YOLOv8-based pretrained model
- Supports various input formats: directory of images
- Configurable confidence threshold for object detection

## Requirements

- Python 3.6 or later


## Usage

1. Clone the repository or download the code files.

2. Install the required dependencies by running `pip install ultralytics`.

3. Prepare your dataset:
   - Place the images to be labeled in a directory.

4- Adjust the script parameters in the `labeling_script.py` file (optional):
- `--destination`: Specify the destination folder for labeled images and labels. (Default:`dataset`)
- `--model_path` : Specify the path to the YOLOv8 model. (Default: `best.pt`)
- `--images_path`: Specify the path to the directory containing the images to be labeled. (Default: `train/`)
- `--conf_threshold`: Set the confidence threshold for object detection. (Default: `0.3`)

5- Run the script by executing `python labeling_script.py` or `python3 labeling_script.py`.

6- The labeled images and corresponding label files will be stored in the specified destination folder.

`Note`: Make sure to have the YOLOv8 model file (`best.pt`) available in the specified path or provide the correct path to your pretrained model.
## Acknowledgements

- This project utilizes the [ultralytics](https://ultralytics.com/) library for YOLOv8 object detection.


## License

[MIT](https://choosealicense.com/licenses/mit/)