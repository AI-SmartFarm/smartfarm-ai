import json
import unittest

from PIL import Image

from helpers import ROOT, load_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.module = load_pipeline()
        self.pipeline = self.module.DiagnosisPipeline.__new__(self.module.DiagnosisPipeline)
        categories = json.loads((ROOT / 'models/rfdetr-s-synthetic-v5/categories.json').read_text(encoding='utf-8'))
        self.pipeline.class_names = {v['id'] - 1: v['name'] for v in categories.values()}
        self.ids = {v: k for k, v in self.pipeline.class_names.items()}
        self.pipeline.knowledge = json.loads((ROOT / 'data/disease_knowledge.json').read_text(encoding='utf-8'))
        self.pipeline.severity_models = {}
        self.pipeline.detector = None

    def diagnose(self, classes, confidences, boxes=None):
        boxes = boxes or [[0, 0, 30, 30] for _ in classes]
        self.module.detect = lambda *args: (boxes, [self.ids[c] for c in classes], confidences)
        return self.pipeline.diagnose(Image.new('RGB', (100, 100)), 'tomato')

    def test_overlapping_normal_does_not_hide_disease(self):
        result = self.diagnose(['tomato_normal', 'tomato_disease18'], [0.95, 0.9])
        self.assertEqual({d['class'] for d in result['detections']}, {'tomato_normal', 'tomato_disease18'})
        self.assertIn('prevention_principles', result['detections'][1]['diagnosis'])

    def test_same_disease_duplicates_keep_highest_confidence(self):
        result = self.diagnose(['tomato_disease18'] * 2, [0.8, 0.9])
        self.assertEqual(len(result['detections']), 1)
        self.assertEqual(result['detections'][0]['confidence'], 0.9)

    def test_distinct_overlapping_diseases_survive(self):
        result = self.diagnose(['tomato_disease18', 'tomato_disease19'], [0.9, 0.8])
        self.assertEqual(len(result['detections']), 2)

    def test_separate_lesions_of_same_class_survive(self):
        result = self.diagnose(['tomato_disease18'] * 2, [0.9, 0.8], [[0, 0, 10, 10], [20, 20, 30, 30]])
        self.assertEqual(len(result['detections']), 2)

    def test_other_crop_is_filtered(self):
        self.assertEqual(self.diagnose(['pepper_disease3'], [0.9])['result'], 'no_detection')

    def test_empty_detection_is_healthy_without_severity(self):
        self.assertEqual(self.diagnose([], [])['result'], 'no_detection')


if __name__ == '__main__':
    unittest.main()
