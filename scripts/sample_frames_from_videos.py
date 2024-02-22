import argparse
import os
import cv2

def extract_frames(video_dir, image_dir, num_frames):
    os.makedirs(image_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error opening video file {video_file}")
                continue

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = length // num_frames

            base_name = os.path.splitext(video_file)[0]

            for i in range(num_frames):
                frame_id = i * interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if ret:
                    output_filename = os.path.join(image_dir, f"{base_name}_{i:03d}.png")
                    cv2.imwrite(output_filename, frame)
                else:
                    print(f"Error reading frame {frame_id} from {video_file}")

            cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video files.")
    parser.add_argument("--video_dir", help="Directory containing the videos")
    parser.add_argument("--image_dir", help="Directory to store the images")
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames to extract")

    args = parser.parse_args()
    extract_frames(args.video_dir, args.image_dir, args.num_frames)
