import asyncio
import io
import threading
import unittest

import httpx
from PIL import Image

from helpers import load_api


class StubPipeline:
    def __init__(self, **kwargs):
        self.calls = []

    def diagnose(self, image, crop, threshold=0.15, tiles=1):
        self.calls.append((crop, threshold, tiles))
        return {'result': 'no_detection', 'crop': crop}


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.api = load_api(StubPipeline)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.api.app, raise_app_exceptions=False), base_url='http://test')
        buffer = io.BytesIO()
        Image.new('RGB', (20, 20)).save(buffer, format='JPEG')
        self.jpeg = buffer.getvalue()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.api._inference_executor.shutdown(wait=True, cancel_futures=True)

    async def diagnose(self, data=None, raw=None, key='test-key'):
        return await self.client.post('/diagnose', headers={'X-API-Key': key},
                                      data=data or {'crop': 'tomato'},
                                      files={'image': ('test.jpg', self.jpeg if raw is None else raw, 'image/jpeg')})

    async def test_auth_health_and_model_version(self):
        self.assertEqual((await self.client.get('/health')).status_code, 200)
        self.assertEqual((await self.client.get('/ping')).status_code, 401)
        self.assertEqual((await self.client.get('/ping', headers={'X-API-Key': 'wrong'})).status_code, 401)
        self.assertEqual((await self.client.get('/ping', headers={'X-API-Key': 'test-key'})).status_code, 200)
        self.assertEqual((await self.diagnose(key='wrong')).status_code, 401)
        result = await self.diagnose()
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['model_version'], self.api.MODEL_VERSION)
        self.assertEqual(self.api.pipeline.calls, [('tomato', 0.15, 1)])

    async def test_invalid_options_never_reach_inference(self):
        for field, value in [('tiles', '0'), ('tiles', '-1'), ('tiles', 'a'),
                             ('threshold', '-1'), ('threshold', '2'), ('threshold', 'nan'),
                             ('threshold', 'inf'), ('threshold', '-inf')]:
            with self.subTest(field=field, value=value):
                result = await self.diagnose({'crop': 'tomato', field: value})
                self.assertEqual(result.status_code, 422)
        self.assertEqual(self.api.pipeline.calls, [])

    async def test_valid_threshold_boundaries_and_tiles(self):
        for threshold in ['0', '1', '0.15']:
            with self.subTest(threshold=threshold):
                result = await self.diagnose({'crop': 'tomato', 'threshold': threshold, 'tiles': '3'})
                self.assertEqual(result.status_code, 200)

    async def test_invalid_images_crop_and_required_fields(self):
        self.assertEqual((await self.diagnose(raw=b'not an image')).status_code, 400)
        self.assertEqual((await self.diagnose(raw=self.jpeg[:20])).status_code, 400)
        self.assertEqual((await self.diagnose({'crop': 'banana'})).status_code, 400)
        self.assertEqual((await self.diagnose({'tiles': '1'})).status_code, 422)
        self.assertEqual(self.api.pipeline.calls, [])

    async def test_inference_error_is_a_server_error_and_next_request_runs(self):
        def fail(*args, **kwargs):
            raise ValueError('model failed')
        original = self.api.pipeline.diagnose
        self.api.pipeline.diagnose = fail
        self.assertEqual((await self.diagnose()).status_code, 500)
        self.api.pipeline.diagnose = original
        self.assertEqual((await asyncio.wait_for(self.diagnose(), 2)).status_code, 200)

    async def test_health_remains_responsive_and_inference_stays_serial(self):
        entered = threading.Event()
        release = threading.Event()
        starts = []
        original = self.api.pipeline.diagnose
        def blocking(*args, **kwargs):
            starts.append(len(starts))
            if len(starts) == 1:
                entered.set()
                if not release.wait(3):
                    raise TimeoutError('test release timed out')
            return original(*args, **kwargs)
        self.api.pipeline.diagnose = blocking
        first = asyncio.create_task(self.diagnose())
        waiting = []
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            # Inference is deliberately blocked; health must finish before release.
            health = await asyncio.wait_for(self.client.get('/health'), 1)
            self.assertEqual(health.status_code, 200)
            # More waiters than the shared HTTP thread limit must not starve ping.
            waiting = [asyncio.create_task(self.diagnose()) for _ in range(45)]
            await asyncio.sleep(0.05)
            self.assertEqual((await asyncio.wait_for(self.client.get('/ping', headers={'X-API-Key': 'test-key'}), 1)).status_code, 200)
            self.assertEqual(len(starts), 1)
        finally:
            release.set()
            responses = await asyncio.gather(first, *waiting)
        self.assertTrue(all(r.status_code == 200 for r in responses))
        self.assertEqual(len(starts), 46)

    async def test_cancelled_request_does_not_overlap_next_inference(self):
        entered = threading.Event()
        release = threading.Event()
        starts = []
        original = self.api.pipeline.diagnose
        def blocking(*args, **kwargs):
            starts.append(len(starts))
            if len(starts) == 1:
                entered.set()
                if not release.wait(3):
                    raise TimeoutError('test release timed out')
            return original(*args, **kwargs)
        self.api.pipeline.diagnose = blocking
        first = asyncio.create_task(self.diagnose())
        second = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(self.diagnose())
            await asyncio.sleep(0.05)
            self.assertEqual(len(starts), 1)
        finally:
            release.set()
            if second:
                response = await second
                self.assertEqual(response.status_code, 200)
            else:
                await asyncio.gather(first, return_exceptions=True)
        self.assertEqual(len(starts), 2)


if __name__ == '__main__':
    unittest.main()
