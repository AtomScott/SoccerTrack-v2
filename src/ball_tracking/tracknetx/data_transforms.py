import random
import numpy as np
import cv2


class RandomCrop:
    """
    Always crops the image to a specified size, with a probability of including the tracked object.

    Assumptions:
        - Input frames are numpy arrays of shape (num_frames * 3, H, W)
        - Input heatmaps are numpy arrays of shape (num_frames, H, W)
        - Input coordinates are numpy arrays of shape (num_frames, 2) where each row is [x, y]
        - Coordinate values are in pixel space (not normalized)
        - All input arrays are properly aligned (same number of frames)
    """

    def __init__(self, crop_height, crop_width, include_object_prob=0.5):
        """
        Initialize the RandomCrop transformation.

        Args:
            crop_height (int): Height of the cropped patch.
            crop_width (int): Width of the cropped patch.
            include_object_prob (float): Probability of including the tracked object in the crop
                                         (must be between 0 and 1).
        """
        if not (0 <= include_object_prob <= 1):
            raise ValueError("include_object_prob must be between 0 and 1")
        if crop_height <= 0 or crop_width <= 0:
            raise ValueError("Crop dimensions must be positive")

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.include_object_prob = include_object_prob

    def __call__(self, frames, heatmaps, coordinates):
        """
        Apply the random crop transformation.

        Args:
            frames (np.ndarray): Concatenated frames of shape (num_frame * 3, H, W).
            heatmaps (np.ndarray): Heatmaps of shape (num_frame, H, W).
            coordinates (np.ndarray): Object coordinates of shape (num_frame, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Transformed frames, heatmaps, and coordinates.

        Raises:
            ValueError: If input shapes are inconsistent or invalid.
        """
        # Input validation
        if (
            len(frames.shape) != 3
            or len(heatmaps.shape) != 3
            or len(coordinates.shape) != 2
        ):
            raise ValueError("Invalid input shapes")

        num_frames = frames.shape[0] // 3
        if heatmaps.shape[0] != num_frames or coordinates.shape[0] != num_frames:
            raise ValueError("Inconsistent number of frames across inputs")

        H, W = frames.shape[1], frames.shape[2]
        if heatmaps.shape[1:] != (H, W):
            raise ValueError("Heatmap dimensions don't match frame dimensions")

        # Decide whether to include the object in the crop
        include_object = random.random() < self.include_object_prob

        if include_object:
            # Find all valid object positions within the image
            valid_indices = [
                i for i, (x, y) in enumerate(coordinates) if 0 <= x < W and 0 <= y < H
            ]

            if valid_indices:
                # Randomly select one valid object position
                selected_idx = random.choice(valid_indices)
                center_x, center_y = coordinates[selected_idx]

                # Determine crop boundaries to include the object
                left_min = max(0, int(center_x - self.crop_width + 1))
                left_max = min(int(center_x), W - self.crop_width)
                top_min = max(0, int(center_y - self.crop_height + 1))
                top_max = min(int(center_y), H - self.crop_height)

                if left_max < left_min:
                    left_min = max(0, int(center_x - self.crop_width / 2))
                    left_max = min(int(center_x), W - self.crop_width)
                if top_max < top_min:
                    top_min = max(0, int(center_y - self.crop_height / 2))
                    top_max = min(int(center_y), H - self.crop_height)

                left = (
                    random.randint(left_min, left_max)
                    if left_max > left_min
                    else left_min
                )
                top = random.randint(top_min, top_max) if top_max > top_min else top_min
            else:
                # Fallback to random crop if no valid positions are found
                left = random.randint(0, max(W - self.crop_width, 0))
                top = random.randint(0, max(H - self.crop_height, 0))
        else:
            # Random crop anywhere
            left = random.randint(0, max(W - self.crop_width, 0))
            top = random.randint(0, max(H - self.crop_height, 0))

        # Perform the crop on frames
        cropped_frames = frames[
            :, top : top + self.crop_height, left : left + self.crop_width
        ]

        # If the cropped area is smaller than the desired crop size, pad the image
        if (
            cropped_frames.shape[1] < self.crop_height
            or cropped_frames.shape[2] < self.crop_width
        ):
            padded_frames = np.zeros(
                (frames.shape[0], self.crop_height, self.crop_width), dtype=frames.dtype
            )
            padded_frames[:, : cropped_frames.shape[1], : cropped_frames.shape[2]] = (
                cropped_frames
            )
            cropped_frames = padded_frames

        # Perform the crop on heatmaps
        cropped_heatmaps = heatmaps[
            :, top : top + self.crop_height, left : left + self.crop_width
        ]

        # Pad heatmaps if necessary
        if (
            cropped_heatmaps.shape[1] < self.crop_height
            or cropped_heatmaps.shape[2] < self.crop_width
        ):
            padded_heatmaps = np.zeros(
                (heatmaps.shape[0], self.crop_height, self.crop_width),
                dtype=heatmaps.dtype,
            )
            padded_heatmaps[
                :, : cropped_heatmaps.shape[1], : cropped_heatmaps.shape[2]
            ] = cropped_heatmaps
            cropped_heatmaps = padded_heatmaps

        # Adjust coordinates
        cropped_coordinates = coordinates.copy()
        cropped_coordinates[:, 0] -= left
        cropped_coordinates[:, 1] -= top

        # Ensure coordinates are within the cropped image
        cropped_coordinates = np.clip(
            cropped_coordinates,
            a_min=0,
            a_max=[self.crop_width - 1, self.crop_height - 1],
        )

        return cropped_frames, cropped_heatmaps, cropped_coordinates


class RandomHorizontalFlip:
    """
    Randomly horizontally flips the frames and heatmaps with a given probability.
    """

    def __init__(self, flip_prob=0.5):
        """
        Initialize the RandomHorizontalFlip transformation.

        Args:
            flip_prob (float): Probability of applying the horizontal flip.
        """
        if not (0 <= flip_prob <= 1):
            raise ValueError("flip_prob must be between 0 and 1")

        self.flip_prob = flip_prob

    def __call__(self, frames, heatmaps, coordinates):
        """
        Apply the random horizontal flip to the frames and heatmaps.

        Args:
            frames (np.ndarray): Concatenated frames of shape (num_frame * 3, H, W).
            heatmaps (np.ndarray): Heatmaps corresponding to frames, shape (num_frame, H, W).
            coordinates (np.ndarray): Coordinates of the object in the frames, shape (num_frame, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Transformed frames, heatmaps, and coordinates.
        """
        if random.random() < self.flip_prob:
            # Flip frames horizontally
            flipped_frames = frames.copy()
            flipped_frames = flipped_frames[:, :, ::-1]

            # Flip heatmaps horizontally
            flipped_heatmaps = heatmaps.copy()
            flipped_heatmaps = flipped_heatmaps[:, :, ::-1]

            # Adjust coordinates
            H, W = frames.shape[1], frames.shape[2]
            flipped_coordinates = coordinates.copy()
            flipped_coordinates[:, 0] = (W - 1) - flipped_coordinates[:, 0]

            return flipped_frames, flipped_heatmaps, flipped_coordinates
        else:
            return frames, heatmaps, coordinates


class Resize:
    """
    Resize transformation for frames and heatmaps.
    """

    def __init__(self, target_height, target_width):
        """
        Initialize the Resize transformation.

        Args:
            target_height (int): Desired image height after resizing.
            target_width (int): Desired image width after resizing.
        """
        if target_height <= 0 or target_width <= 0:
            raise ValueError("Target dimensions must be positive integers.")
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, frames, heatmaps, coordinates):
        """
        Apply resizing to frames and heatmaps.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Resized frames, heatmaps, and adjusted coordinates.
        """
        num_frames = frames.shape[0] // 3
        original_height, original_width = frames.shape[1], frames.shape[2]
        height_ratio = self.target_height / original_height
        width_ratio = self.target_width / original_width

        # Resize frames
        resized_frames = []
        for i in range(num_frames):
            frame = frames[i * 3 : (i + 1) * 3].transpose(1, 2, 0)  # (H, W, C)
            frame_resized = cv2.resize(frame, (self.target_width, self.target_height))
            frame_resized = frame_resized.transpose(2, 0, 1)  # (C, H, W)
            resized_frames.append(frame_resized)
        resized_frames = np.concatenate(resized_frames, axis=0)  # (num_frame * 3, H, W)

        # Resize heatmaps
        resized_heatmaps = []
        for i in range(num_frames):
            heatmap = heatmaps[i]
            heatmap_resized = cv2.resize(
                heatmap, (self.target_width, self.target_height)
            )
            resized_heatmaps.append(heatmap_resized)
        resized_heatmaps = np.stack(resized_heatmaps, axis=0)  # (num_frame, H, W)

        # Adjust coordinates
        resized_coordinates = coordinates.copy()
        resized_coordinates[:, 0] *= width_ratio
        resized_coordinates[:, 1] *= height_ratio

        return resized_frames, resized_heatmaps, resized_coordinates
