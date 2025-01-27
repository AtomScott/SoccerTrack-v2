# 950セットの訓練データセットから100, 500セットの訓練データセットを作るためのコード
# 950セットの中からランダムに選ばれる

import os
import shutil
import numpy as np
from pathlib import Path

# 元のフォルダと出力フォルダのパスを指定
train_dir = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/ball_tracking_dataset-stride-1/train")
output_dir = Path("/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/ball_tracking_dataset-stride-1/train_500")

# 必要なフォルダ構造を作成
output_frames_dir = output_dir / "frames"
output_frames_dir.mkdir(parents=True, exist_ok=True)

# npyファイルを読み込む
coordinates = np.load(train_dir / "coordinates.npy", allow_pickle=True)
sequences = np.load(train_dir / "sequences.npy", allow_pickle=True)
visibility = np.load(train_dir / "visibility.npy", allow_pickle=True)

# データのセット数を取得
num_sets = coordinates.shape[0]

# 100セットをランダムに選択
random_indices = np.random.choice(num_sets, 100, replace=False)

# 選択したデータを新しいフォルダに保存
new_coordinates = coordinates[random_indices]
new_sequences = sequences[random_indices]
new_visibility = visibility[random_indices]

# 新しい npy ファイルを保存
np.save(output_dir / "coordinates.npy", new_coordinates)
np.save(output_dir / "sequences.npy", new_sequences)
np.save(output_dir / "visibility.npy", new_visibility)

# 対応するフレームをコピー
for sequence in new_sequences:
    for frame_path in sequence:
        src_frame = Path(frame_path)
        dst_frame = output_frames_dir / src_frame.name
        if not dst_frame.exists():
            shutil.copy(src_frame, dst_frame)

print("Random selection and saving complete!")
