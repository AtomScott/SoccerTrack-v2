# 正しくデータセットが作れているか確認するコード
# labelsに入っている位置情報をimagesにプロットする

import os
import cv2
import random
from pathlib import Path

# 画像とラベルのディレクトリ
images_dir = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/yolo_dataset/images/test")
labels_dir = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/yolo_dataset/labels/test")

# 出力先ディレクトリ（プロット結果保存用）
output_dir = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/YOLO/plot_exp")
output_dir.mkdir(parents=True, exist_ok=True)

# ディレクトリ内の全画像ファイルを取得し、ランダムに10枚選択
all_images = list(images_dir.glob("*.jpg"))
if len(all_images) < 10:
    print("十分な画像ファイルがありません。")
    selected_images = all_images
else:
    selected_images = random.sample(all_images, 10)

for img_path in selected_images:
    label_path = labels_dir / (img_path.stem + ".txt")
    
    # 画像を読み込み
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"画像が見つかりません: {img_path}")
        continue
    
    height, width = img.shape[:2]
    
    # ラベルファイルが存在すれば座標を読み込む
    if label_path.is_file():
        with open(label_path, 'r') as f:
            lines = f.read().strip().splitlines()
        
        for line in lines:
            if not line:
                continue
            parts = line.split()
            # YOLO形式: class_id x_center_norm y_center_norm width_norm height_norm
            if len(parts) != 5:
                continue
            
            _, x_center_norm, y_center_norm, _, _ = parts
            x_center_norm = float(x_center_norm)
            y_center_norm = float(y_center_norm)
            
            # 正規化座標をピクセル座標に変換
            x_center = int(x_center_norm * width)
            y_center = int(y_center_norm * height)
            
            # 座標に円を描く
            cv2.circle(img, (x_center, y_center), radius=5, color=(0, 0, 255), thickness=2)
    else:
        print(f"ラベルファイルが見つかりません: {label_path}")
    
    # 出力ファイル名を作成
    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), img)
    print(f"保存しました: {output_path}")
