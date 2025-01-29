"""Frame sampling strategies for video processing."""

from abc import ABC, abstractmethod
import numpy as np


class FrameSampler(ABC):
    """Abstract base class for frame sampling strategies."""

    @abstractmethod
    def should_sample(self, frame_idx: int) -> bool:
        """Determine if the frame should be sampled."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the sampler's internal state."""
        pass


class IntervalSampler(FrameSampler):
    """Sample frames at fixed intervals."""

    def __init__(self, interval: int = 1):
        """
        Initialize interval sampler.

        Args:
            interval: Number of frames to skip between samples.
        """
        if interval < 1:
            raise ValueError("Interval must be >= 1")
        self.interval = interval

    def should_sample(self, frame_idx: int) -> bool:
        """Return True if the frame should be sampled based on the interval."""
        return frame_idx % self.interval == 0

    def reset(self) -> None:
        """Reset sampler state (no-op for interval sampler)."""
        pass


class UncertaintySampler(FrameSampler):
    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold

    def should_sample(self, frame_idx: int) -> bool:
        # Implement uncertainty-based sampling logic
        prediction = self.model.predict(frame_idx)
        uncertainty = self.calculate_uncertainty(prediction)
        return uncertainty > self.threshold

    def reset(self) -> None:
        # Reset any internal state if needed
        pass
