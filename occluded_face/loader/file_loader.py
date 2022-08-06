from __future__ import annotations

import os

import cv2
import numpy as np

from occluded_face.dataclass import File


class FileLoader:

    def __init__(self, image_dir: str, imshow: function | None = None):
        self._image_dir = image_dir

        image_files = []
        for file in os.listdir(self._image_dir):
            if file.endswith(".jpg"):
                image_files.append(file)

        self._index = 0
        self._image_files = image_files
        self._num_images = len(image_files)

        self.imshow = imshow if imshow else cv2.imshow

    def _load(self) -> File:
        index = self._index
        file_name = self._image_files[index]
        path = os.path.join(self._image_dir, file_name)
        
        file = File(file_name, cv2.imread(path), *self._get_labels(file_name))
        
        self._index = index + 1
        
        return file

    def _get_labels(self, file_name: str) -> tuple[bool, bool, bool]:
        """
            File name format:
            <is_occluded>_<is_top_occluded>_<is_bottom_occluded>XXXXXXX
        """
        labels = file_name.split('_')[:3]

        is_occluded = labels[0] == '1'
        is_top_occluded = labels[1] == '1'
        is_bottom_occluded = labels[2][0] == '1'

        return is_occluded, is_top_occluded, is_bottom_occluded

    def aggregate(self) -> list[dict]:
        return [file.info for file in self]

    def __iter__(self) -> "FileLoader":
        self._index = 0
        return self

    def __next__(self) -> File:
        if self._index == self._num_images:
            raise StopIteration

        file = self._load()
        return file
