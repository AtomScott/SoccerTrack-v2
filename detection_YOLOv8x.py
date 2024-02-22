from IPython import display
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image

from roboflow import Roboflow

def main():

    # Load a model
    model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

    # model.to("cuda")

    # Train the model
    model.train(data="dataset.yaml", epochs=500, imgsz=640, batch=8, workers=4, degrees=90.0, device="cuda:0")
    
    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category

    # Test the model
    results = model.predict()

    # Export the model
    model.export(format="torchscript")

if __name__ == "__main__":
    main()