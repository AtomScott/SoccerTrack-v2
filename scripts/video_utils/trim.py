import argparse
import moviepy.editor as mp

def trim_video(input_video_path, output_video_path, start_time, end_time):
    """
    Trims a video to a specified start and end time.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str): The path where the trimmed video will be saved.
        start_time (float): The start time in seconds to trim the video from.
        end_time (float): The end time in seconds to trim the video to.
    """
    video = mp.VideoFileClip(input_video_path).subclip(start_time, end_time)
    video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

def main():
    parser = argparse.ArgumentParser(description='Trim a video to a given start and end time.')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('output', help='Output video file path')
    parser.add_argument('start_time', type=float, help='Start time in seconds')
    parser.add_argument('end_time', type=float, help='End time in seconds')

    args = parser.parse_args()

    trim_video(args.input, args.output, args.start_time, args.end_time)

if __name__ == "__main__":
    main()

# Example usage:
# python trim.py input.mp4 output.mp4 10 20