from eyewitness.dataset_util import BboxDataSet
from eyewitness.evaluation import BboxMAPEvaluator

from naive_detector import get_person_detector


if __name__ == '__main__':
    dataset_folder = 'VOCdevkit/VOC2007'
    # dataset_folder = '/home/stream/Pictures/MAS'
    dataset_VOC_2007 = BboxDataSet(dataset_folder, 'VOC2007')
    object_detector = get_person_detector('MobileNet')
    bbox_map_evaluator = BboxMAPEvaluator(test_set_only=False)
    # the hog detector for person AP only 0.006
    # the opencv2 haar Cascade for person AP only 0.0006
    # the mobile net with VOC2007 with mAP 0.53
    print(bbox_map_evaluator.evaluate(object_detector, dataset_VOC_2007)['mAP'])
