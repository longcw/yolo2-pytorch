from collections import Counter, defaultdict
import os
import re
from torch.utils import data

from datasets.utils import gteaannots

"""
Action labels are composed of a vert and a set of actions such as
('pour', ['honey, bread']), they are then processed to string labels
such as 'pour honey bread'

About the labels
two action labels have only 1 and 2 occurences in dataset:
('stir', ['cup']) and ('put', ['tea'])
by removing them we get to 71 action classes

If we want to be as close as possible to initial results
we have to upse the original cvpr labels and process
some object so as to regroup them ('cupPlateBowl' and
'spoonForkKnife classes')
==> 44 action classes

self.classes is computed and contains (action, (object1, ...)) tuples
Not all frames are annotated !
"""


class GTEAGazePlus(data.Dataset):
    def __init__(self, root_folder="data/GTEAGazePlus",
                 original_labels=True,
                 seqs=['Ahmad', 'Alireza', 'Carlos',
                       'Rahul', 'Shaghayegh', 'Yin'],
                 use_flow=False):

        self.cvpr_labels = ['open_fridge', 'close_fridge',
                            'put_cupPlateBowl',
                            'put_spoonForkKnife_cupPlateBowl',
                            'take_spoonForkKnife_cupPlateBowl',
                            'take_cupPlateBowl', 'take_spoonForkKnife',
                            'put_lettuce_cupPlateBowl',
                            'read_recipe', 'take_plastic_spatula',
                            'open_freezer', 'close_freezer',
                            'put_plastic_spatula',
                            'cut_tomato_spoonForkKnife',
                            'put_spoonForkKnife',
                            'take_tomato_cupPlateBowl',
                            'turnon_tap', 'turnoff_tap',
                            'take_cupPlateBowl_plate_container',
                            'turnoff_burner', 'turnon_burner',
                            'cut_pepper_spoonForkKnife',
                            'put_tomato_cupPlateBowl',
                            'put_milk_container', 'put_oil_container',
                            'take_oil_container', 'close_oil_container',
                            'open_oil_container', 'take_lettuce_container',
                            'take_milk_container', 'open_fridge_drawer',
                            'put_lettuce_container', 'close_fridge_drawer',
                            'compress_sandwich',
                            'pour_oil_oil_container_skillet',
                            'take_bread_bread_container',
                            'cut_mushroom_spoonForkKnife',
                            'put_bread_cupPlateBowl', 'put_honey_container',
                            'take_honey_container', 'open_microwave',
                            'crack_egg_cupPlateBowl',
                            'open_bread_container', 'open_honey_container']
        # Label tags
        self.original_labels = original_labels
        self.untransform = None  # Needed for visualizer

        self.path = root_folder
        self.label_path = os.path.join(self.path, 'labels_cleaned')
        self.all_seqs = ['Ahmad', 'Alireza', 'Carlos',
                         'Rahul', 'Yin', 'Shaghayegh']

        self.seqs = seqs
        self.use_flow = use_flow

        # Compute classes
        if self.original_labels:
            self.classes = self.get_cvpr_classes()
        else:
            self.classes = self.get_repeat_classes(2)

        # Sanity check on computed class nb
        if self.original_labels:
            self.class_nb = len(self.cvpr_labels)
        else:
            self.class_nb = 32

        assert len(self.classes) == self.class_nb,\
            "{0} classes found, should be {1}".format(
                len(self.classes), self.class_nb)

    def get_subj_classes(self, seqs=None):
        """Returns a list of label lists where the label lists are
        grouped by subject
        [[(action, obj, b, e), ... ] for subject 1, [],...]
        """
        annot_paths = [os.path.join(self.label_path, annot_file)
                       for annot_file in os.listdir(self.label_path)]
        subjects_classes = []

        # Get classes for each subject
        if seqs is None:
            subjects = self.all_seqs
        else:
            subjects = seqs

        for subject in subjects:
            subject_annot_files = [filepath for filepath in annot_paths
                                   if subject in filepath]

            # Process files to extract action_lists
            subject_lines = [gteaannots.process_lines(subject_file)
                             for subject_file in subject_annot_files]

            # Flatten actions for each subject
            subject_labels = [label for sequence_labels in subject_lines
                              for label in sequence_labels]

            subjects_classes.append(subject_labels)
        return subjects_classes

    def get_repeat_classes(self, repetition_per_subj=2, seqs=None):
        """
        Gets the classes that are repeated at least
        repetition_per_subject times for each subject
        """
        subjects_classes = self.get_subj_classes(seqs=seqs)
        repeated_subjects_classes = []
        for subj_labels in subjects_classes:
            subj_classes = self.get_repeated_annots(subj_labels,
                                                    repetition_per_subj)

            repeated_subjects_classes.append(subj_classes)
        shared_classes = []
        # Get classes present at least twice for the subject
        first_subject_classes = repeated_subjects_classes.pop()
        for subject_class in first_subject_classes:
            shared = all(subject_class in subject_classes
                         for subject_classes in repeated_subjects_classes)
            if shared:
                shared_classes.append(subject_class)
        return sorted(shared_classes)

    def get_repeated_annots(self, annot_lines, repetitions):
        """
        Given list of annotations in format [action, objects, begin, end]
        returns list of (action, objects) tuples for the (action, objects)
        that appear at least repetitions time
        """
        action_labels = [self.get_class_str(act, self.original_label_transform(obj))
                         for (act, obj, b, e) in annot_lines]
        counted_labels = defaultdict(int)
        for label in action_labels:
            counted_labels[label] += 1
        repeated_labels = [label for label, count in counted_labels.items()
                           if count >= repetitions]
        return repeated_labels

    def get_cvpr_classes(self, seqs=None):
        """Gets original cvpr classes as list of classes as
        list of strings
        """
        subjects_classes = self.get_subj_classes(seqs=seqs)
        all_classes = []
        for subj_labels in subjects_classes:
            # Remove internal spaces
            for (action, obj, b, e) in subj_labels:
                action_str = self.get_class_str(action, obj)
                if action_str in self.cvpr_labels:
                    all_classes.append((action,
                                        self.original_label_transform(obj)))
        return sorted(list(set(all_classes)))

    def original_label_transform(self, objects):
        mutual_1 = ['fork', 'knife', 'spoon']
        mutual_2 = ['cup', 'plate', 'bowl']
        processed_obj = []
        for obj in objects:
            if obj in mutual_1:
                processed_obj.append('spoonForkKnife')
            elif obj in mutual_2:
                processed_obj.append('cupPlateBowl')
            else:
                processed_obj.append(obj)
        return tuple(processed_obj)

    def get_class_str(self, action, objects):
        """Transforms action and objects inputs
        """
        if self.original_labels:
            objects = self.original_label_transform(objects)
        action_str = '_'.join((action.replace(' ', ''),
                               '_'.join(objects)))
        return action_str

    def get_all_actions(self, action_object_classes):
        """Extracts all possible actions in the format (action,
        objects, subject, recipe, first_frame, last_frame) """
        annot_paths = [os.path.join(self.label_path, annot_file)
                       for annot_file in os.listdir(self.label_path)]
        actions = []
        # Get classes for each subject
        for subject in self.seqs:
            subject_annot_files = [filepath for filepath in annot_paths
                                   if subject in filepath]
            for annot_file in subject_annot_files:
                recipe = re.search('.*_(.*).txt', annot_file).group(1)
                action_lines = gteaannots.process_lines(annot_file)
                for action, objects, begin, end in action_lines:
                    if self.original_labels:
                        objects = self.original_label_transform(objects)
                    if (action, objects) in action_object_classes:
                        actions.append((action, objects, subject, recipe,
                                        begin, end))
        return actions

    def get_action_counts(self, actions):
        c = Counter(actions)
        counts = []
        for classe in self.classes:
            counts.append(c[classe])
        return counts
