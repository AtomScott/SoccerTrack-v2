import argparse
import subprocess


def trim_video(input_video_path, output_video_path, start_time, end_time):
    """
    Trims a video to a specified start and end time using ffmpeg.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str): The path where the trimmed video will be saved.
        start_time (float): The start time in seconds to trim the video from.
        end_time (float): The end time in seconds to trim the video to.
    """
    duration = end_time - start_time
    cmd = [
        'ffmpeg', '-i', input_video_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',  # Use stream copy for faster processing
        output_video_path,
        '-y'  # Overwrite output file if it exists
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Trim a video to a given start and end time.")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", help="Output video file path")
    parser.add_argument("start_time", type=float, help="Start time in seconds")
    parser.add_argument("end_time", type=float, help="End time in seconds")

    args = parser.parse_args()

    trim_video(args.input, args.output, args.start_time, args.end_time)


if __name__ == "__main__":
    main()

# Example usage:
# python trim.py input.mp4 output.mp4 10 20
