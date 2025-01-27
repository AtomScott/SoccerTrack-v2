from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from predict_yolov8_dist import evaluate_model, load_test_data

def main():
    # モデルの読み込み（必要ならランダム初期化する場合はweights=Noneを指定）
    model = YOLO('/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/YOLO/yolov8n.pt')
    
    # 学習・評価に関する設定
    data_yaml = 'data.yaml'
    total_epochs = 100
    imgsz = (3250, 500)
    rect = True
    batch = 8
    project = 'YOLOBallDetection'
    name = 'yolov8n_stride-1_frame-950'
    
    # 検証データのディレクトリ設定
    images_dir = "/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/yolo_dataset-stride-1/images/val"
    labels_dir = "/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/SoccerTrack-v2/data/interim/yolo_dataset-stride-1/labels/val"
    output_dir = "/home/nakamura/desktop/playbox/ball_detection/TrackNetV3/YOLO/plot_exp/train950"
    os.makedirs(output_dir, exist_ok=True)
    
    # 異なる距離閾値セットの定義
    distance_thresholds = [5, 4, 3, 2, 1]
    distance_thresholds_20 = [20, 10, 5, 2, 1]
    
    # 検証データの読み込み
    image_paths, label_paths = load_test_data(images_dir, labels_dir)
    
    # mAP 記録用リストの初期化
    epochs = []
    map5px_list = []
    map5to1px_list = []
    map20px_list = []
    map20to1px_list = []
    
    # エポックごとの学習と評価
    for epoch in range(1, total_epochs + 1):
        print(f"Epoch {epoch} / {total_epochs}")
        
        # 1 エポック分の学習を実行
        model.train(
            data=data_yaml,
            epochs=1,             
            imgsz=imgsz,
            rect=rect,
            batch=batch,
            project=project,
            name=name,
            exist_ok=True         
        )
        
        # distance_thresholds に基づく評価
        results = evaluate_model(
            model,
            image_paths,
            label_paths,
            output_dir,
            distance_thresholds
        )
        
        # distance_thresholds_20 に基づく評価
        results_20 = evaluate_model(
            model,
            image_paths,
            label_paths,
            output_dir,
            distance_thresholds_20
        )
        
        # mAP データの抽出・記録
        epochs.append(epoch)
        map5px_list.append(results.get("mAP@5px", 0.0))
        map5to1px_list.append(results.get("mAP@5~1px", 0.0))
        map20px_list.append(results_20.get("mAP@5px", 0.0))      # ここでは結果キーを使い回していますが、実際は"mAP@20px"等で返却されることを想定
        map20to1px_list.append(results_20.get("mAP@5~1px", 0.0)) # 同様に調整が必要

    # distance_thresholds に基づくグラフの描画・保存
    plt.figure()
    plt.plot(epochs, map5px_list, label='mAP@5px')
    plt.plot(epochs, map5to1px_list, label='mAP@5~1px')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP over Epochs (thres=5)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mAP_over_epochs_thres5.png"))
    plt.close()
    
    # distance_thresholds_20 に基づくグラフの描画・保存
    plt.figure()
    plt.plot(epochs, map20px_list, label='mAP@20px')
    plt.plot(epochs, map20to1px_list, label='mAP@20~1px')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP over Epochs (thres=20)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mAP_over_epochs_thres20.png"))
    plt.close()
    
    print("Training and evaluation completed. Graphs saved.")
    
if __name__ == "__main__":
    main()
