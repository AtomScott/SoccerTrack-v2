# テストデータに対してYOLOモデルの推論・評価を行うコード
# 3連続フレームの中央のフレームに対して評価
# 距離ベースのmAPを計算
# 予測結果を描画した画像をoutput_dirへ保存

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

############################
# 1) ユーティリティ関数
############################

def load_yolo_model(weights_path):
    """
    YOLOモデル(ultralytics)をロード
    """
    model = YOLO(weights_path)
    return model

def load_test_data_middle(images_dir, labels_dir, stride=3):
    """
    画像・ラベルパスを取得し、strideフレームごとに真ん中のフレームを選択
    デフォルトでは3フレームごとに真ん中のフレームを選択
    """
    all_image_paths = sorted(Path(images_dir).glob("*.jpg"))
    all_label_paths = sorted(Path(labels_dir).glob("*.txt"))
    
    # strideフレームごとに真ん中のフレームを抽出
    # 例: stride=3 → 各3フレームセットから2番目のフレームを選択
    image_paths = all_image_paths[stride//2::stride]  # 中央フレーム
    label_paths = all_label_paths[stride//2::stride]  # 同じくラベルも
    return image_paths, label_paths

def parse_yolo_label(label_path):
    """
    YOLO形式のラベルを読み込み (cls, x_center, y_center, width, height)
    正規化された座標
    """
    with open(label_path, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # 不完全な行をスキップ
        cls, x_center, y_center, width, height = map(float, parts[:5])
        labels.append((cls, x_center, y_center, width, height))
    return labels

def yolo_to_bbox(x_center, y_center, w, h, img_w, img_h):
    """
    YOLO形式(正規化座標) → バウンディングボックス(x1, y1, x2, y2)
    """
    x1 = int(max((x_center - w / 2) * img_w, 0))
    y1 = int(max((y_center - h / 2) * img_h, 0))
    x2 = int(min((x_center + w / 2) * img_w, img_w - 1))
    y2 = int(min((y_center + h / 2) * img_h, img_h - 1))
    return (x1, y1, x2, y2)

def center_of_box(box):
    """
    box: (x1, y1, x2, y2)
    """
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return (cx, cy)

def is_center_in_8x8(center, gt_center, side_len=8):
    """
    "予測中心"が "GTを中心とする8x8" の中にあるかどうか
    """
    half = side_len / 2
    x1 = gt_center[0] - half
    y1 = gt_center[1] - half
    x2 = gt_center[0] + half
    y2 = gt_center[1] + half
    return (x1 <= center[0] <= x2) and (y1 <= center[1] <= y2)

def draw_predictions_and_ground_truth(img, pred_boxes, gt_boxes, detection_status, side_len=8):
    """
    画像に予測ボックス（緑）と正解ボックス（赤）を描画し、
    左上にGround Truth、Predictionの中心座標と検出結果を表示
    """
    # ボックス描画
    for box in pred_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)  # 緑色
    for box in gt_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)  # 赤色

    # 各ボックスの中心を描画
    for gt_box in gt_boxes:
        gt_center = center_of_box(gt_box)
        cv2.circle(img, (int(gt_center[0]), int(gt_center[1])), 3, (0, 0, 255), -1)  # 赤色
    for pred_box in pred_boxes:
        pred_center = center_of_box(pred_box)
        cv2.circle(img, (int(pred_center[0]), int(pred_center[1])), 3, (0, 255, 0), -1)  # 緑色

    # 左上にテキスト表示
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    x_text = 10
    y_text = 30
    y_offset = 25

    # 正解座標と予測座標をリスト化
    gt_text = "Ground Truth: " + ", ".join([f"({int(cx)}, {int(cy)})" for (cx, cy) in [center_of_box(box) for box in gt_boxes]])
    pred_text = "Prediction    : " + ", ".join([f"({int(cx)}, {int(cy)})" for (cx, cy) in [center_of_box(box) for box in pred_boxes]])
    detection_text = "Detection: Success" if detection_status else "Detection: Failure"

    cv2.putText(img, gt_text, (x_text, y_text), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, pred_text, (x_text, y_text + y_offset), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, detection_text, (x_text, y_text + 2 * y_offset), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

    return img

def calculate_iou(box1, box2):
    """
    box1, box2: (x1, y1, x2, y2)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_w = max(0, x2_inter - x1_inter)
    inter_h = max(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

############################
# 2) 評価関数
############################

def evaluate_model(model, image_paths, label_paths, output_dir, distance_thresholds):
    """
    - Mean Euclidean Distance / MSE
    - Precision/Recall: 予測中心がGT中心の8x8ボックス内にあるかどうか
    - mAP@5px, mAP@10px, ..., mAP@1px: 距離閾値を用いたmAP計算
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    distances = []
    distances_sq = []

    # 距離ベースのmAP用設定
    distance_thresholds = sorted(distance_thresholds, reverse=True)  # 高い閾値から
    dist_tp = {th: 0 for th in distance_thresholds}
    dist_fp = {th: 0 for th in distance_thresholds}
    individual_precisions = {}
    prec_values = []

    total_frames = len(image_paths)  # 真ん中のフレームのみの個数

    for idx, (img_path, lbl_path) in enumerate(tqdm(zip(image_paths, label_paths), total=total_frames, desc="Evaluating")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue
        H, W, _ = img.shape

        # GT読み込み
        gt_data = parse_yolo_label(lbl_path)
        gt_boxes = []
        for (cls, xc, yc, ww, hh) in gt_data:
            box = yolo_to_bbox(xc, yc, ww, hh, W, H)
            gt_boxes.append(box)

        # 推論
        results = model(img)
        if not results or not results[0].boxes:
            pred_boxes = []
        else:
            pred_xywhn = results[0].boxes.xywhn.cpu().numpy()
            pred_boxes = []
            for (xc, yc, bw, bh) in pred_xywhn:
                box = yolo_to_bbox(xc, yc, bw, bh, W, H)
                pred_boxes.append(box)

        # (中心ベース) Precision/Recall計算
        matched_gt = set()
        image_detection_success = False
        for pbox in pred_boxes:
            p_center = center_of_box(pbox)
            matched = False
            for i, gt_box in enumerate(gt_boxes):
                gt_center = center_of_box(gt_box)
                if is_center_in_8x8(p_center, gt_center, side_len=8):
                    if i not in matched_gt:
                        matched_gt.add(i)
                        d = np.sqrt((p_center[0] - gt_center[0])**2 + (p_center[1] - gt_center[1])**2)
                        distances.append(d)
                        distances_sq.append(d**2)
                        true_positives += 1
                        matched = True
                        image_detection_success = True
                        break
            if not matched:
                false_positives += 1

        fn_local = len(gt_boxes) - len(matched_gt)
        false_negatives += fn_local

        # 距離ベースmAP更新
        for pbox in pred_boxes:
            p_center = center_of_box(pbox)
            for gt_box in gt_boxes:
                gt_center = center_of_box(gt_box)
                # 各予測と正解の中心間の距離を計算
                dist = np.sqrt((p_center[0] - gt_center[0])**2 + (p_center[1] - gt_center[1])**2)
                # 距離閾値に基づいてTP/FPをカウント
                for th in distance_thresholds:
                    if dist <= th:
                        dist_tp[th] += 1
                    else:
                        dist_fp[th] += 1

        # 可視化のために画像に描画
        vis_img = img.copy()
        vis_img = draw_predictions_and_ground_truth(vis_img, pred_boxes, gt_boxes, image_detection_success)
        save_path = output_dir / f"{Path(img_path).stem}_vis.jpg"
        cv2.imwrite(str(save_path), vis_img)

    # Precision/Recall (中心ベース)
    precision_center = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall_center = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # 平均距離 / MSE
    mean_dist = np.mean(distances) if distances else 0.0
    mean_mse = np.mean(distances_sq) if distances_sq else 0.0

    # 距離ベースmAP計算
    map_results = {}
    for th in distance_thresholds:
        tp_ = dist_tp[th]
        fp_ = dist_fp[th]
        prec_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        individual_precisions[th] = prec_
        prec_values.append(prec_)
        if th == distance_thresholds[0]:
            map_results[f"mAP@5px"] = prec_
    map_results["mAP@5~1px"] = np.mean(prec_values) if prec_values else 0.0

    return {
        "mean_distance": mean_dist,
        "mean_squared_error": mean_mse,
        "precision": precision_center,
        "recall": recall_center,
        "mAP@5px": map_results.get("mAP@5px", 0.0),
        "mAP@10px": individual_precisions.get(10, 0.0),
        "mAP@5~1px": map_results.get("mAP@5~1px", 0.0),
        "individual_precisions": individual_precisions
    }

############################
# 3) メイン関数
############################

def main():
    # データパス (75フレーム想定)
    images_dir = "/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/yolo_dataset-stride-1/images/test"
    labels_dir = "/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/yolo_dataset-stride-1/labels/test"
    weights_path = "/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/YOLO/YOLOBallDetection/yolov8n_stride-1_frame-100/weights/best.pt"
    output_dir = "/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/YOLO/plot_exp"

    # モデル読み込み
    model = load_yolo_model(weights_path)

    # データロード（stride=3で真ん中のフレームのみ選択）
    image_paths, label_paths = load_test_data_middle(images_dir, labels_dir, stride=3)

    # mAP5px or mAP20px
    # distance_thresholds=[5, 4, 3, 2, 1]
    distance_thresholds=[20, 10, 5, 2, 1]
    # 評価
    results = evaluate_model(model, image_paths, label_paths, output_dir, distance_thresholds)

    # 結果出力
    print("YOLO Evaluation:")
    print(f"Mean Euclidean Distance: {results['mean_distance']:.2f} px")
    print(f"Mean Squared Error: {results['mean_squared_error']:.2f} px^2")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"mAP@{distance_thresholds[0]}px: {results['mAP@5px']:.4f}")
    print(f"mAP@{distance_thresholds[0]}~{distance_thresholds[4]}px: {results['mAP@5~1px']:.4f}")
    print("Individual Precisions by threshold:")
    for th, prec in sorted(results['individual_precisions'].items(), reverse=True):
        print(f"  {th}px: {prec:.4f}")

if __name__ == "__main__":
    main()
