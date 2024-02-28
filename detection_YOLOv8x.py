from IPython import display
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image

from roboflow import Roboflow

# 学習データ数
train_dataset = 50

# 学習かテストか
train = True

def main():

    # train and val
    if train:

        # Load a model
        model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

        # Train the model
        model.train(data="dataset_" + str(train_dataset) + ".yaml", epochs=1, imgsz=32, batch=4, workers=4, degrees=90.0, device="cuda:0")

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        map = metrics.box.maps   # a list contains map50-95 of each category
        print(map)
        print(type(map))

        # Export the model
        # model.export(format="torchscript")

    # test
    else:
    
        # Load a best model
        model = YOLO('C:\\Users\\黒田堅仁\\OneDrive - 筑波大学\\ドキュメント\\GitHub\\SoccerTrack-v2\\runs\\detect\\train4\\weights\\best.pt')
        
        # test the model
        # ret = model("/kaggle/input/car-object-detection/data/testing_images",save=True, conf=0.2, iou=0.5)

if __name__ == "__main__":
    main()