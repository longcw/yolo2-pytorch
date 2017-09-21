import os
import re


def process_lines(annot_path, inclusion_condition=None):
    """
    Returns list of action_object as
    ["action_name", "object1, object2", first_frame, last_frame]
    """
    with open(annot_path) as f:
        lines = f.readlines()
    processed_lines = []
    for line in lines:
        matches = re.search('<(.*)><(.*)> \((.*)-(.*)\)', line)
        if matches:
            action_label, object_label = matches.group(1), matches.group(2)
            begin, end = int(matches.group(3)), int(matches.group(4))
            object_labels = tuple(object_label.split(','))
            if inclusion_condition is None or\
                    inclusion_condition(action_label, object_labels):
                processed_lines.append((action_label, object_labels,
                                        begin, end))
    return processed_lines


def _class_string(action, objects):
    return action + ' ' + ' '.join(objects)


def process_annots(processed_lines, class_string=_class_string):
    """
    Returns a dictionnary with frame as key and
    value 'action object1 object2 object3' from
    the gtea annotation text file

    :param class_string: function that transforms action and label
    into action_label class string
    """
    # create annotation_dict
    annot_dict = {}
    for action, object_label, begin, end in processed_lines:
        for frame in range(begin, end + 1):
            annot_dict[frame] = class_string(action, object_label)
    return annot_dict


def get_all_classes(label_path, inclusion_condition=None,
                    class_string=_class_string,
                    no_action_label=True):
    """
    Returns list of all action_object classes

    :param class_string: function that transforms action and label
    into action_label class string
    """
    sequences = os.listdir(label_path)
    seqs = [os.path.join(label_path, seq) for seq in sequences]
    object_actions = []
    for seq in seqs:
        annots = process_lines(seq, inclusion_condition)
        for annot in annots:
            object_actions.append((annot[0:2]))
    unique_object_actions = sorted(list(set(object_actions)))
    unique_classes = [class_string(action, objects)
                      for action, objects in unique_object_actions]
    if no_action_label:
        unique_classes.append('None')
    return unique_classes
