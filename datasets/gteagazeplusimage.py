import random
import os

import numpy as np

from datasets.utils import loader, visualize
from datasets.gteagazeplus import GTEAGazePlus


class GTEAGazePlusImage(GTEAGazePlus):
    def __init__(self, root_folder="data/GTEAGazePlus",
                 original_labels=True, seqs=['Ahmad', 'Alireza', 'Carlos',
                                             'Rahul', 'Shaghayegh', 'Yin'],
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
                         original_labels=original_labels,
                         seqs=seqs)

        # Set video params
        self.base_transform = base_transform
        self.transform = transform
        self.untransform = untransform
        self.rgb_path = os.path.join(self.path, 'png')
        self.video_path = os.path.join(self.path, 'avi_files')

        self.action_clips = self.get_all_actions(self.classes)
        # Remove actions that are too short
        action_labels = [(action, obj) for (action, obj, subj, rec,
                                            beg, end) in self.action_clips]
        assert len(action_labels) > 100
        self.class_counts = self.get_action_counts(action_labels)
        assert sum(self.class_counts) == len(action_labels)

    def __getitem__(self, index):
        # Load clip
        action, objects, subject, recipe, beg, end = self.action_clips[index]
        sequence_name = subject + '_' + recipe

        # One hot encoding
        annot = np.zeros(self.class_nb)
        class_idx = self.classes.index((action, objects))
        annot[class_idx] = 1

        frame_idx = random.randint(beg, end)
        frame_name = "{frame:010d}.png".format(frame=frame_idx)
        img_path = os.path.join(self.rgb_path, sequence_name, frame_name)
        img = loader.load_rgb_image(img_path)

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)
        return img, annot

    def __len__(self):
        return len(self.action_clips)

    def plot_hist(self):
        """Plots histogram of action classes as sampled in self.action_clips
        """
        labels = [self.get_class_str(action, obj)
                  for (action, obj, subj, rec, beg, end) in self.action_clips]
        visualize.plot_hist(labels, proportion=True)

    def get_class_items(self, index, frame_nb=None):
        """Get all images of an action clip as list of image
        Args:
            frame_nb(int): if indicates, number of frames to sample uniformly
                in each action clap, if not present, frames are sampled densely
        """
        # Load clip info
        action, objects, subject, recipe, beg, end = self.action_clips[index]
        sequence_name = subject + '_' + recipe

        # Get class index
        class_idx = self.classes.index((action, objects))

        # Return list of action tensors
        imgs = []

        if frame_nb is None:
            frame_idxs = range(beg, end)
        else:
            frame_idxs = np.linspace(beg, end, frame_nb)
            frame_idxs = [int(frame_idx) for frame_idx in frame_idxs]

        for frame_idx in frame_idxs:
            frame_name = "{frame:010d}.png".format(frame=frame_idx)
            img_path = os.path.join(self.rgb_path,
                                    sequence_name, frame_name)
            img = loader.load_rgb_image(img_path)
            if self.base_transform is not None:
                img = self.base_transform(img)
            imgs.append(img)
        return imgs, class_idx
