python scripts/yolov8/train_val.py \
  --input ./datasets/roboflow-split/2024-02-28-trainsize10-randomstate1/data.yaml \
  --output ./outputs/det_metrics_dict_2024-02-28-trainsize10-randomstate1.json \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device cuda:0 