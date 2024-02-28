import argparse
from ultralytics import YOLO
import json
from loguru import logger

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
    det_metrics_dict = det_metrics.results_dict

    logger.info(det_metrics_dict)

    # Save det_metrics_dict to a JSON file
    with open(args.output, 'w') as file:
        json.dump(det_metrics_dict, file)

if __name__ == "__main__":
    main()

# Example script usage:
# python detection_YOLOv8x.py --train_dataset 1 --epochs 1 --imgsz 32 --batch 4 --workers 4 --device cuda:0 --output results_json\det_metrics_dict_1.json