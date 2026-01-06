import numpy as np
import argparse
from pathlib import Path
from sportslabkit.logger import logger
from sportslabkit.camera.calibrate import find_intrinsic_camera_parameters

# Create the parser
parser = argparse.ArgumentParser(description='Calibrate camera using a checkerboard pattern.')

# Add the arguments
parser.add_argument('--checkerboard_mp4_path', type=str, help='The path to the checkerboard mp4 file')
parser.add_argument('--points_to_use', type=int, help='The number of points to use')

# Parse the arguments
args = parser.parse_args()
checkerboard_mp4_path = Path(args.checkerboard_mp4_path)

# Find intrinsic camera parameters
K, D, mapx, mapy = find_intrinsic_camera_parameters(
    checkerboard_mp4_path,
    fps=1,
    scale=1,
    draw_on_save=True,
    points_to_use=args.points_to_use,
    calibration_method="fisheye"
)

logger.info(f"mapx: {mapx}, mapy: {mapy}")

# Save the mappings to files
save_path = checkerboard_mp4_path.parent
np.save(save_path / 'mapx.npy', mapx)
np.save(save_path / 'mapy.npy', mapy)

logger.info(f"Saved mapx and mapy to {save_path}")
