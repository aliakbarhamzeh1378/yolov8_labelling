import os
import shutil
import argparse
from ultralytics import YOLO

def label_objects(destination, model_path, images_path, conf_threshold):
    # Set the destination paths for images and labels
    image_destination = os.path.join(destination, 'image')
    label_destination = os.path.join(destination, 'label')

    # Create destination directories if they don't exist
    os.makedirs(destination, exist_ok=True)
    os.makedirs(image_destination, exist_ok=True)
    os.makedirs(label_destination, exist_ok=True)

    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Predict objects in the given images
    results = model.predict(source=images_path, conf=conf_threshold, stream=True)

    # Iterate over the results
    for i in results:
        if len(i.boxes.cls) == 0:
            continue
        name = os.path.split(i.path)[-1]
        path = os.path.dirname(i.path)
        label_path = os.path.join(path, name.replace(name.split('.')[-1], 'txt'))

        # Create a label file and write object information
        with open(label_path, "w") as f:
            for andis, _ in enumerate(i.boxes.cls):
                cords = list(i.boxes.xywhn)
                x, y, w, h = cords[andis]
                cls = i.boxes.cls[andis]
                f.write(f"{int(cls)} {float(x)} {float(y)} {float(w)} {float(h)}\n")

        # Move the label file and the image to the destination directories
        shutil.move(label_path, label_destination)
        shutil.move(os.path.join(path, name), image_destination)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='YOLOv8 Object Labeling')
    parser.add_argument('--destination', type=str, default='dataset', help='Destination folder')
    parser.add_argument('--model_path', type=str, default='best.pt', help='Path to the YOLOv8 model')
    parser.add_argument('--images_path', type=str, default='train/cigarette', help='Path to the images')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='Confidence threshold for object detection')

    args = parser.parse_args()

    # Call the label_objects function with the provided arguments
    label_objects(args.destination, args.model_path, args.images_path, args.conf_threshold)
