from IPython import display
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image
import json
import sys


def main():

    # the number of the train dataset
    args = sys.argv
    # first argument is the number of the train dataset
    train_dataset = args[1]
    
    # Load a model
    model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data="datasets\dataset_" + str(train_dataset) + ".yaml", epochs=1000, imgsz=1280, batch=4, workers=4, degrees=90.0, device="cuda:0")
    
    # Validate the model
    det_metrics = model.val()  # no arguments needed, dataset and settings remembered
    det_metrics_dict = det_metrics.results_dict

    print(det_metrics_dict)

    # Save det_metrics_dict to a JSON file
    with open('det_metrics_dict' + str(train_dataset) + '.json', 'w') as file:
        json.dump(det_metrics_dict, file)


if __name__ == "__main__":
    main()