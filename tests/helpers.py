import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, replacements):
    spec = importlib.util.spec_from_file_location('test_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


def load_pipeline():
    # Tensor/model imports are unused by these postprocessing tests.
    torchvision = types.ModuleType('torchvision')
    torchvision.models = types.SimpleNamespace()
    torchvision.transforms = types.SimpleNamespace()
    ops = types.ModuleType('torchvision.ops')
    ops.batched_nms = None
    rfdetr = types.ModuleType('rfdetr')
    rfdetr.RFDETRSmall = None
    return load_script('diagnosis_pipeline', {
        'torch': types.ModuleType('torch'), 'torchvision': torchvision,
        'torchvision.ops': ops, 'rfdetr': rfdetr,
    })


def load_api(pipeline_class):
    pipeline = types.ModuleType('diagnosis_pipeline')
    pipeline.CROPS = ['pepper', 'strawberry', 'lettuce', 'cucumber', 'tomato']
    pipeline.DiagnosisPipeline = pipeline_class
    with patch.dict('os.environ', {'API_KEY': 'test-key'}):
        return load_script('api', {'diagnosis_pipeline': pipeline})
