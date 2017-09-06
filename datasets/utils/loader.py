import cv2
import os
import numpy as np
import pickle
from PIL import Image


def load_rgb_image(path):
    """
    loads image from path
    :param path: absolute or relative path to file
    :rtype: PIL Image
    :return: RGB Image
    """
    image = Image.open(path)
    return image.convert("RGB")


def load_depth_image(path):
    """
    loads depth image from path
    :param path: absolute or relative path to depth image
    :rtype: PIL Image
    :return: Depth image
    """
    image = Image.open(path)
    return image


class OpenCvError(Exception):
    pass


def get_video_capture(file_name):
    """
    Returns opencv video_capture name based on file_name
    """
    file_found = os.path.isfile(file_name)

    # Check video exists as file
    if not file_found:
        raise OpenCvError('Video file {0} doesn\'t exist'.format(
            file_name))
    video_capture = cv2.VideoCapture(file_name)

    # Check video could be read
    if not video_capture.isOpened():
        raise OpenCvError('Video is not opened')

    return video_capture


def get_clip(video_capture, frame_begin, frame_nb):
    """ Returns clip of video as list of numpy.ndarrays
    of dimensions [channels, frames, height, width]

    Args:
        video_capture (cv2.VideoCapture): opencv videoCapture
            object loading a video
        frame_begin (int): first frame from clip
        frame_nb (int): number of frames to retrieve
    """

    # Get video dimensions
    if video_capture.isOpened():
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    else:
        raise OpenCvError('Video is not opened')

    # Retrieve successive frames starting from frame_begin frame
    video_capture.set(1, frame_begin)

    clip = []
    # Fill clip array with consecutive frames
    for frame_idx in range(frame_nb):
        flag, frame = video_capture.read()
        if not flag:
            raise OpenCvError('Could not read frame {0}'.format(
                frame_idx + frame_begin))

        frame = frame[:, :, ::-1]
        clip.append(frame)
    return clip


def get_stacked_frames(image_folder, frame_begin, frame_nb,
                       frame_template="{frame:010d}.png",
                       use_open_cv=True, to_numpy=False):
    """ Returns numpy array/PIL image of stacked images with dimensions
    [channels, frames, height, width]

    Args:
        image_folder (str): folder containing the images
            in format frame_template
        frame_nb (int): number of consecutive frames to stack
        use_open_cv (bool): wheter to use opencv or PIl image reader
            if PIL is used, PIL image are returned
    """

    clip = []
    for idx in range(frame_nb):
        frame_idx = frame_begin + idx
        image_path = os.path.join(image_folder,
                                  frame_template.format(frame=frame_idx))
        if use_open_cv:
            img = cv2.imread(image_path)
            if img is None:
                raise OpenCvError('Could not open\
                                  image {0}'.format(image_path))
            img = img[:, :, ::-1]
        else:
            img = Image.open(image_path)
            if to_numpy:
                img = np.asarray(img)
        clip.append(img)

    return clip


def get_stacked_flow_arrays(image_folder, frame_begin, frame_nb,
                            flow_x_template="{frame:05d}_x.jpg",
                            flow_y_template="{frame:05d}_y.jpg",
                            minmax_filename=None):
    """
    Retrives flow data from folder where x and y flows are
    stored separately as jpg files with values normalized
    to [0, 255]
    """
    clip = []
    # Retrieve minmax dict in format {frame_idx: [min_x, max_x, min_y, max_y]}
    if minmax_filename is not None:
        with open(os.path.join(image_folder,
                               minmax_filename), 'rb') as minmax_file:
            minmax = pickle.load(minmax_file)
    for idx in range(frame_nb):
        frame_idx = frame_begin + idx
        flow_x_path = os.path.join(image_folder,
                                   flow_x_template.format(frame=frame_idx))
        flow_y_path = os.path.join(image_folder,
                                   flow_y_template.format(frame=frame_idx))
        img_flow_x = Image.open(flow_x_path)
        img_flow_y = Image.open(flow_y_path)
        flow_x = np.asarray(img_flow_x).astype(np.float32)
        flow_y = np.asarray(img_flow_y).astype(np.float32)
        if minmax_filename is not None:
            frame_minmax = minmax[frame_idx]

            flow_x = cv2.normalize(flow_x, flow_x,
                                   alpha=frame_minmax[0],
                                   beta=frame_minmax[1],
                                   norm_type=cv2.NORM_MINMAX)
            flow_y = cv2.normalize(flow_y, flow_y,
                                   alpha=frame_minmax[2],
                                   beta=frame_minmax[3],
                                   norm_type=cv2.NORM_MINMAX)

        flow = np.stack((flow_x, flow_y), axis=2)
        clip.append(flow)
    return clip


def get_stacked_numpy_arrays(image_folder, frame_begin, frame_nb,
                             frame_template="{frame:05d}.pickle"):
    """ Returns list of stacked numpy arrays with dimensions
    by reading files created by np.save from folder image_folder

    Args:
        image_folder (str): folder containing the images in format
            in format frame_template
        frame_nb (int): number of consecutive frames to stack
        use_open_cv (bool): wheter to use opencv or PIl image reader
            if PIL is used, PIL image are returned
    """

    clip = []
    for idx in range(frame_nb):
        frame_idx = frame_begin + idx
        image_path = os.path.join(image_folder,
                                  frame_template.format(frame=frame_idx))
        img = np.load(image_path)
        clip.append(img)

    return clip


def format_img_from_opencv(img_array):
    """
    Transforms an opencv numpy array [width, height, BRG]
    to [RGB, width, height]
    """
    frame_rgb = img_array[:, :, ::-1]
    arranged_frame = np.rollaxis(frame_rgb, 2, 0)
    return arranged_frame
