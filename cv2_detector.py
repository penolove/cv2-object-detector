import arrow
import cv2
import numpy as np

from eyewitness.detection_utils import DetectionResult
from eyewitness.config import BoundedBoxObject
from eyewitness.image_id import ImageId
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import ImageHandler, Image


class HogPersonDetectorWrapper(ObjectDetector):
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image_obj) -> DetectionResult:
        frame = np.array(image_obj.pil_image_obj)
        rects, weights = self.hog.detectMultiScale(
            frame, winStride=(3, 3), padding=(16, 16), scale=1.05)

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

        picked_idx = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        detected_objects = []
        for idx in picked_idx:
            x1, y1, x2, y2 = rects[idx]
            score = weights[idx][0]
            if score < self.threshold:
                continue
            detected_objects.append(BoundedBoxObject(x1, y1, x2, y2, 'person', score, ''))

        image_dict = {
            'image_id': image_obj.image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)

        return detection_result

    @property
    def valid_labels(self):
        return set(['person'])


class CascadeClassifierPersonWrapper(ObjectDetector):
    def __init__(self):
        self.person_cascade = cv2.CascadeClassifier('cascade_classifier/haarcascade_fullbody.xml')
        self.resize = None

    def detect(self, image_obj) -> DetectionResult:
        frame = np.array(image_obj.pil_image_obj)
        if self.resize is not None:
            frame = cv2.resize(frame, self.resize)
            ori_height, ori_width = frame.shape[:2]
            x_scale_up_ratio = self.resize[0] / ori_width
            y_scale_up_ratio = self.resize[1] / ori_height
        else:
            x_scale_up_ratio = 1
            y_scale_up_ratio = 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        persons = self.person_cascade.detectMultiScale(gray, 1.1, 1)
        persons = np.array([[x, y, x + w, y + h] for (x, y, w, h) in persons])
        picked_idx = non_max_suppression(persons, probs=None, overlapThresh=0.65)
        detected_objects = []
        for idx in picked_idx:
            x1, y1, x2, y2 = persons[idx]
            x1 = x1 / x_scale_up_ratio
            x2 = x2 / x_scale_up_ratio
            y1 = y1 / y_scale_up_ratio
            y2 = y2 / y_scale_up_ratio
            detected_objects.append(BoundedBoxObject(x1, y1, x2, y2, 'person', 0, ''))

        image_dict = {
            'image_id': image_obj.image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)

        return detection_result

    @property
    def valid_labels(self):
        return set(['person'])


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    return pick

