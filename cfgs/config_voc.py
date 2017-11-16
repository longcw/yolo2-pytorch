import numpy as np


# trained model
h5_fname = 'yolo-voc.weights.h5'

# VOC
label_names =('stop',
              'speedLimitUrdbl',
              'speedLimit25',
              'pedestrianCrossing',
              'speedLimit35',
              'turnLeft',
              'slow',
              'speedLimit15',
              'speedLimit45',
              'rightLaneMustTurn',
              'signalAhead',
              'keepRight',
              'laneEnds',
              'school',
              'merge',
              'addedLane',
              'rampSpeedAdvisory40',
              'rampSpeedAdvisory45',
              'curveRight',
              'speedLimit65',
              'truckSpeedLimit55',
              'thruMergeLeft',
              'speedLimit30',
              'stopAhead',
              'yield',
              'thruMergeRight',
              'dip',
              'schoolSpeedLimit25',
              'thruTrafficMergeLeft',
              'noRightTurn',
              'rampSpeedAdvisory35',
              'curveLeft',
              'rampSpeedAdvisory20',
              'noLeftTurn',
              'zoneAhead25',
              'zoneAhead45',
              'doNotEnter',
              'yieldAhead',
              'roundabout',
              'turnRight',
              'speedLimit50',
              'rampSpeedAdvisoryUrdbl',
              'rampSpeedAdvisory50',
              'speedLimit40',
              'speedLimit55',
              'doNotPass',
              'intersection'
             )
num_classes = len(label_names)

anchors = np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)], dtype=np.float)
num_anchors = len(anchors)

