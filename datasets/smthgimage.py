import random
import os

import numpy as np

from datasets.utils import loader
from datasets.smthg import Smthg


class SmthgImage(Smthg):
    def __init__(self, root_folder="data/smthg-smthg", split='train',
                 transform=None, untransform=None, base_transform=None):
        """
        Args:
            transform: transformations to apply during training
            base_transform: transformations to apply during testing
            untransform: transform to reapply after transformation
                to visualize original image
            use_video (bool): whether to use video inputs or png inputs
        """
        super().__init__(root_folder=root_folder,
                         split=split)

        # Set image params
        self.base_transform = base_transform
        self.transform = transform
        self.untransform = untransform

    def __getitem__(self, index):
        # Load clip
        clip_id, label, max_frame = self.sample_list[index]

        # One hot encoding
        annot = np.zeros(self.class_nb)
        class_idx = self.classes.index(label)
        annot[class_idx] = 1

        frame_idx = random.randint(1, max_frame)
        frame_name = self.frame_template.format(frame=frame_idx)
        img_path = os.path.join(self.path_from_id(clip_id), frame_name)
        img = loader.load_rgb_image(img_path)

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)
        return img, annot

    def get_class_items(self, index, frame_nb=None):
        """Get all images of an action clip as list of image
        Args:
            frame_nb(int): if indicates, number of frames to sample uniformly
                in each action clap, if not present, frames are sampled densely
        """
        # Load clip info
        clip_id, label, max_frame = self.sample_list[index]

        # Get class index
        if self.split == 'test':
            class_idx = 0
        else:
            class_idx = self.classes.index(label)

        # Return list of action tensors
        imgs = []

        if frame_nb is None:
            frame_idxs = range(max_frame)
        else:
            # Sample frame_nb frames uniformly in all clip frames
            frame_idxs = np.linspace(1, max_frame, frame_nb)
            frame_idxs = [int(frame_idx) for frame_idx in frame_idxs]

        for frame_idx in frame_idxs:
            frame_name = self.frame_template.format(frame=frame_idx)
            img_path = os.path.join(self.path_from_id(clip_id), frame_name)
            img = loader.load_rgb_image(img_path)
            if self.base_transform is not None:
                img = self.base_transform(img)
            imgs.append(img)
        return imgs, class_idx
