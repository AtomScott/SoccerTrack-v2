# TrackNetの評価のためのコード
# 距離ベースのmAPを求める
# 予測結果を描画した画像をoutput_dirへ保存

import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

from model import TrackNetXModel  # 環境に合わせて調整

############################
# 1) ユーティリティ
############################

def load_tracknet_model(checkpoint_path: str, device: str = "cpu"):
    """TrackNetの学習済みモデルをロード"""
    model = TrackNetXModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model

def preprocess_frames(frames):
    """TrackNetへの入力形式に合わせた前処理"""
    processed = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = frame.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        processed.append(frame)
    input_tensor = np.concatenate(processed, axis=0)  # (C*3, H, W)
    return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)

def compute_iou_tracknet(pred_center, gt_center, side_len=8):
    """TrackNetは点予測なので、点を8×8の正方形とみなしてIoUを計算"""
    px, py = pred_center
    gx, gy = gt_center
    pred_box = [px - side_len/2, py - side_len/2, px + side_len/2, py + side_len/2]
    gt_box   = [gx - side_len/2, gy - side_len/2, gx + side_len/2, gy + side_len/2]
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area   = (gt_box[2] - gt_box[0])   * (gt_box[3] - gt_box[1])
    union_area = pred_area + gt_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def is_center_in_box(center, box_center, side_len=8):
    """予測中心が正解ボックス内にあるかどうかをチェック"""
    half = side_len / 2
    bx1 = box_center[0] - half
    by1 = box_center[1] - half
    bx2 = box_center[0] + half
    by2 = box_center[1] + half
    return (bx1 <= center[0] <= bx2) and (by1 <= center[1] <= by2)

def draw_predictions_and_ground_truth(img, pred_center, gt_center, detection_status, side_len=8):
    """
    画像に予測中心と正解中心を表示し、8x8ボックスを描画し、左上に座標と検出結果を表示
    """
    half = side_len / 2
    # 予測ボックス
    pred_box = [int(pred_center[0] - half), int(pred_center[1] - half),
                int(pred_center[0] + half), int(pred_center[1] + half)]
    # 正解ボックス
    gt_box = [int(gt_center[0] - half), int(gt_center[1] - half),
              int(gt_center[0] + half), int(gt_center[1] + half)]

    # ボックス描画
    cv2.rectangle(img, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 255, 0), 2)  # 緑色
    cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 255), 2)  # 赤色

    # 各ボックスの中心を描画
    cv2.circle(img, (int(pred_center[0]), int(pred_center[1])), 3, (0, 255, 0), -1)  # 緑色
    cv2.circle(img, (int(gt_center[0]), int(gt_center[1])), 3, (0, 0, 255), -1)  # 赤色

    # テキスト描画
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    x_text = 10
    y_text = 30
    y_offset = 30

    gt_text = f"Ground Truth: ({int(gt_center[0])}, {int(gt_center[1])})"
    pred_text = f"Prediction    : ({int(pred_center[0])}, {int(pred_center[1])})"
    detection_text = "Detection: Success" if detection_status else "Detection: Failure"

    cv2.putText(img, gt_text, (x_text, y_text), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, pred_text, (x_text, y_text + y_offset), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, detection_text, (x_text, y_text + 2*y_offset), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

    return img

############################
# 2) TrackNet評価関数
############################

def evaluate_tracknet_model(sequences, coordinates, visibility, model, output_dir, device, distance_thresholds):
    """
    - 平均ユークリッド距離, MSE
    - Precision, Recall: 「予測中心が正解 8×8 box内にあるかどうか」
    - mAP@5px, mAP@4px, mAP@3px, mAP@2px, mAP@1px: 距離閾値を用いたmAP計算
    - 中央フレームに予測結果を描画して保存
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 中心ベース Precision/Recall
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # ユークリッド距離
    distances = []
    distances_sq = []

    # 距離ベースのmAP用設定
    distance_thresholds = sorted(distance_thresholds, reverse=True)  # 大きい順にソート
    dist_tp = {th: 0 for th in distance_thresholds}
    dist_fp = {th: 0 for th in distance_thresholds}
    individual_precisions = {}
    prec_values = []

    total_frames = len(sequences)

    for idx, sequence in enumerate(tqdm(sequences, total=total_frames, desc="Evaluating sequences")):
        frames = [cv2.imread(str(frame_path)) for frame_path in sequence]
        if any(frame is None for frame in frames):
            print(f"Warning: Unable to read all frames in sequence {idx}. Skipping.")
            continue
        input_tensor = preprocess_frames(frames).to(device)

        # 推論
        with torch.no_grad():
            output = model(input_tensor)
            output_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()

        # 中央フレームのヒートマップから予測中心を取得
        pred_heatmap = output_prob[1]  # assuming channel 1 is for prediction
        pred_y, pred_x = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
        pred_center = (pred_x, pred_y)

        # 正解座標取得
        gt_x, gt_y = coordinates[idx, 1]  # assuming [sequence_index, frame_index, coordinates]
        gt_center = (gt_x, gt_y)
        vis = visibility[idx, 1]  # assuming visibility is [sequence_index, frame_index]

        # 距離計算
        dist = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        distances.append(dist)
        distances_sq.append(dist**2)

        # Precision / Recall (中心ベース)
        if vis:
            if is_center_in_box(pred_center, gt_center, side_len=8):
                true_positives += 1
                detection_success = True
            else:
                false_positives += 1
                detection_success = False
        else:
            false_negatives += 1
            detection_success = False

        # 距離ベースmAP更新
        if vis:
            for th in distance_thresholds:
                if dist <= th:
                    dist_tp[th] += 1
                else:
                    dist_fp[th] += 1

        # 可視化のために中央フレームに描画
        middle_frame_idx = len(sequence) // 2
        original_img = frames[middle_frame_idx].copy()
        vis_img = draw_predictions_and_ground_truth(original_img, pred_center, gt_center, detection_success, side_len=8)
        save_path = output_dir / f"{Path(sequence[middle_frame_idx]).stem}_vis.jpg"
        cv2.imwrite(str(save_path), vis_img)

    # Precision/Recall 計算
    precision_center = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall_center = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # 平均距離 / MSE
    mean_dist = np.mean(distances) if distances else 0.0
    mean_mse  = np.mean(distances_sq) if distances_sq else 0.0

    # 距離ベースmAP計算
    for th in distance_thresholds:
        tp_ = dist_tp[th]
        fp_ = dist_fp[th]
        prec_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        individual_precisions[th] = prec_
        prec_values.append(prec_)
        if th == distance_thresholds[0]:
            map_5px = prec_
    map_5_95 = np.mean(prec_values) if prec_values else 0.0

    return {
        "MeanEuclidianDistance": mean_dist,
        "MSE": mean_mse,
        "Precision_center": precision_center,
        "Recall_center": recall_center,
        # "mAP@5px": individual_precisions.get(5, 0.0),
        # "mAP@4px": individual_precisions.get(4, 0.0),
        # "mAP@3px": individual_precisions.get(3, 0.0),
        # "mAP@2px": individual_precisions.get(2, 0.0),
        # "mAP@1px": individual_precisions.get(1, 0.0),
        # "mAP@5~1px": map_5_95,
        "mAP@20px": individual_precisions.get(20, 0.0),
        "mAP@10px": individual_precisions.get(10, 0.0),
        "mAP@5px": individual_precisions.get(5, 0.0),
        "mAP@2px": individual_precisions.get(2, 0.0),
        "mAP@1px": individual_precisions.get(1, 0.0),
        "mAP@5~1px": map_5_95,
        "individual_precisions": individual_precisions  # 各閾値のPrecisionを追加
    }

############################
# 3) メイン関数
############################

def main():
    # データパス (シーケンス数に合わせて調整)
    sequences_path = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/ball_tracking_dataset-stride-1/test/sequences.npy")
    coordinates_path = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/ball_tracking_dataset-stride-1/test/coordinates.npy")
    visibility_path = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/ball_tracking_dataset-stride-1/test/visibility.npy")
    checkpoint_path = "/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/tracknetx/SoccerTrack-v2/src/ball_tracking/tracknetx/exp-stride=1-weighted_msee/train_100_model-epoch=482-val_total_loss=0.00.ckpt"
    output_dir = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/tracknetx/SoccerTrack-v2/src/ball_tracking/tracknetx/exp")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # データ読込
    sequences = np.load(sequences_path, allow_pickle=True)
    coordinates = np.load(coordinates_path, allow_pickle=True)
    visibility = np.load(visibility_path, allow_pickle=True)

    # モデル読込
    model = load_tracknet_model(checkpoint_path, device)

    # 距離に基づくmAP計算のための閾値設定（例として5px,4px,3px,2px,1px）
    # distance_thresholds = [5, 4, 3, 2, 1]
    distance_thresholds = [20, 10, 5, 2, 1]

    # 評価実行
    results = evaluate_tracknet_model(
        sequences, 
        coordinates, 
        visibility, 
        model, 
        output_dir, 
        device, 
        distance_thresholds
    )

    # 出力
    print("TrackNet Evaluation:")
    print(f"Mean Euclidean Distance: {results['MeanEuclidianDistance']:.2f} px")
    print(f"Mean Squared Error: {results['MSE']:.2f} px^2")
    print(f"Precision(center-based): {results['Precision_center']:.4f}")
    print(f"Recall(center-based): {results['Recall_center']:.4f}")
    # print(f"mAP@5px: {results['mAP@5px']:.4f}")
    # print(f"mAP@4px: {results['mAP@4px']:.4f}")
    # print(f"mAP@3px: {results['mAP@3px']:.4f}")
    # print(f"mAP@2px: {results['mAP@2px']:.4f}")
    # print(f"mAP@1px: {results['mAP@1px']:.4f}")
    # print(f"mAP@5~1px: {results['mAP@5~1px']:.4f}")
    print(f"mAP@20px: {results['mAP@20px']:.4f}")
    print(f"mAP@10px: {results['mAP@10px']:.4f}")
    print(f"mAP@5px: {results['mAP@5px']:.4f}")
    print(f"mAP@2px: {results['mAP@2px']:.4f}")
    print(f"mAP@1px: {results['mAP@1px']:.4f}")
    print(f"mAP@20~1px: {results['mAP@5~1px']:.4f}")
    print("Individual Precisions by threshold:")
    for th, prec in sorted(results['individual_precisions'].items(), reverse=True):
        print(f"  {th}px: {prec:.4f}")

if __name__ == "__main__":
    main()
