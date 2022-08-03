from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class File:
    name: str
    image: np.ndarray
    is_occluded: bool
    is_top_occluded: bool
    is_bottom_occluded: bool

    def show(self):
        cv2.imshow('Image', self.image)

    @property
    def info(self):
        return {
            'is_occluded': self.is_occluded,
            'is_top_occluded': self.is_top_occluded,
            'is_bottom_occluded': self.is_bottom_occluded
        }

    def __repr__(self):
        return f"({self.is_occluded}, {self.is_top_occluded}, {self.is_bottom_occluded})"
    
    def __str__(self):
        return f"({self.is_occluded}, {self.is_top_occluded}, {self.is_bottom_occluded})"