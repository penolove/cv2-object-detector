import arrow
import cv2
import numpy as np

from eyewitness.detection_utils import DetectionResult
from eyewitness.config import BoundedBoxObject
from eyewitness.image_id import ImageId
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import ImageHandler, Image

from cv2_detector import CascadeClassifierPersonWrapper, HogPersonDetectorWrapper

def get_person_detector(model):
    if model == 'Cascade':
        detector = CascadeClassifierPersonWrapper()
    elif model == 'Hog':
        detector = detector = CascadeClassifierPersonWrapper()()
    else:
        raise Exception('not implement error')
    return detector


if __name__ == '__main__':
    object_detector = get_person_detector()
    raw_image_path = 'demo/test_image.jpg'
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    image_obj = Image(image_id, raw_image_path=raw_image_path)
    detection_result = object_detector.detect(image_obj)
    print("detected %s objects" % len(detection_result.detected_objects))
    ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(image_obj.pil_image_obj, "detected_image/drawn_image.jpg")
