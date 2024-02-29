import argparse
from ultralytics import YOLO
import json
from loguru import logger
import numpy as np

def parse_args():
    """
    Parse command line arguments.

    Returns:
        Namespace: The arguments namespace.
    """
    parser = argparse.ArgumentParser(description='Train YOLOv8x model on a specified dataset.')
    parser.add_argument('--input', type=str, required=True, help='Input dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file for saving metrics')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load a model
    model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data=args.input, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device)
    
    # Validate the model
    det_metrics = model.val()

    # all class results
    det_metrics_dict = det_metrics.results_dict

    # each class results
    det_metrics_box = det_metrics.box
    class_output_np = np.full((2,4),0)
    class_output_np[:, 0] = det_metrics_box.p
    class_output_np[:, 1] = det_metrics_box.r
    class_output_np[:, 2] = det_metrics_box.ap50
    class_output_np[:, 3] = det_metrics_box.maps

    # name list of validation indexes
    index_name_list = ['Precision', 'Recall', 'mAP50', 'mAP50-95']

    # remove key not to use
    del det_metrics_dict['fitness']

    # change the key names of all class dict
    all_class_dict = dict()
    for i, key in enumerate(det_metrics_dict.keys()):
        all_class_dict[index_name_list[i]] = float(det_metrics_dict[key])


    # person class to dict
    person_class_dict = dict()
    for i in range(4):
        person_class_dict[index_name_list[i]] = float(class_output_np[0, i])


    # sports_ball class to dict
    ball_class_dict = dict()
    for i in range(4):
        ball_class_dict[index_name_list[i]] = float(class_output_np[1, i])


    # unity each class dict
    results_json_dict = {
        "all":all_class_dict,
        "person":person_class_dict,
        "ball":ball_class_dict
        }


    # Save dict to a JSON file
    with open(args.output, 'w') as file:
        json.dump(results_json_dict, file)


if __name__ == "__main__":
    main()

# Example script usage:
# python detection_YOLOv8x.py --train_dataset 1 --epochs 1 --imgsz 32 --batch 4 --workers 4 --device cuda:0 --output results_json\det_metrics_dict_1.json