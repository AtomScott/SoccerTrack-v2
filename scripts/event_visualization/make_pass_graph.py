import matplotlib.pyplot as plt
import json
import argparse
import pandas as pd
import os

def parse_arguments():
    """
    Parse command line arguments for match ID and event class count.

    Returns:
        argparse.Namespace: Parsed command line arguments containing match_id and num_class.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_id', help="Match ID for the video and annotation files")
    parser.add_argument('--num_class', help="Number of event classes, e.g., '12' or '14'")
    return parser.parse_args()

def main():
    # Example usage
    args = parse_arguments()
    match_ids = [str(match_id) for match_id in args.match_id.split(",")]
    num_class = str(args.num_class)
    
    for match_id in match_ids:
        labels_path = f'data/raw/{match_id}/{match_id}_{num_class}_class_events.json'
        detections_path = f'data/interim/event_detection_tracking/{match_id}/{match_id}_event_detection.json'
        output_graph_path = f'data/interim/event_visualization/{match_id}/{match_id}_compare_pass_prediction_graph.jpg'

        # ファイルを読み込み
        with open(labels_path, 'r') as f:
            labels_df = json.load(f)

        with open(detections_path, 'r') as f:
            detections_df = json.load(f)
        
        make_time_series_graph(labels_df, detections_df, output_graph_path)

def make_time_series_graph(labels_df: pd.DataFrame, detections_df: pd.DataFrame, output_graph_path: str) -> None:
    """
    Generate a time series graph comparing true and predicted PASS events
    and save the graph to the specified path.

    Args:
        labels_df (pd.DataFrame): DataFrame containing true event labels.
        detections_df (pd.DataFrame): DataFrame containing predicted event labels.
        output_graph_path (str): Path to save the generated graph.
    """
    # Extract positions of PASS events
    labels = labels_df['annotations']
    detections = detections_df['predictions']
    true_PASS_positions = extract_pass_positions(labels)
    predicted_PASS_positions = extract_pass_positions(detections)
    true_OUT_positions = extract_out_positions(labels)
    predicted_OUT_positions = extract_out_positions(detections)
    print(len(true_PASS_positions), len(predicted_PASS_positions))
    # Prepare data for the graph
    all_positions = sorted(set(true_PASS_positions + predicted_PASS_positions))

    bar_PASS_width = 400
    bar_OUT_width = 800
    # offset = bar_width / 2

    plt.figure(figsize=(20, 8))
    plt.bar(
        [pos for pos in true_PASS_positions], 
        1, 
        color="blue", 
        width=bar_PASS_width, 
        label="True PASS", 
        align="center"
    )
    plt.bar(
        [pos for pos in predicted_PASS_positions], 
        1, 
        color="red", 
        width=bar_PASS_width, 
        label="Predicted PASS", 
        align="center"
    )
    plt.bar(
        [pos for pos in true_OUT_positions], 
        1, 
        color="orange", 
        width=bar_OUT_width, 
        label="True OUT", 
        align="center"
    )
    plt.bar(
        [pos for pos in predicted_OUT_positions], 
        1, 
        color="green", 
        width=bar_OUT_width, 
        label="Predicted OUT", 
        align="center"
    )
    plt.xlim(min(all_positions) - bar_PASS_width, max(all_positions) + bar_PASS_width)
    # Generate ticks every 10,000
    x_min = min(all_positions)
    x_max = max(all_positions)
    x_ticks = range(x_min - (x_min % 100000), x_max + 100000, 100000)
    plt.xticks(x_ticks)
    # Add labels and title
    plt.xlabel("Time (position)")
    plt.ylabel("PASS & OUT Event (1: Present, 0: Absent)")
    plt.title("Comparison of True and Predicted PASS & OUT Events")
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=1.0)
    plt.tight_layout()
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_graph_path), exist_ok=True)

    # Save the graph to the specified path
    try:
        plt.savefig(output_graph_path, format="jpg", dpi=300)
        print(f"Graph saved successfully at {output_graph_path}")
    except Exception as e:
        print(f"Failed to save graph: {e}")
    plt.close()
    print(f"Saving graph to: {output_graph_path}")

# PASSラベルのposition値を抽出
def extract_pass_positions(data):
    return [int(item["position"]) for item in data if (item["label"] == "PASS")]

# OUTラベルのposition値を抽出
def extract_out_positions(data):
    return [int(item["position"]) for item in data if (item["label"] == "OUT")]

if __name__ == '__main__':
    main()