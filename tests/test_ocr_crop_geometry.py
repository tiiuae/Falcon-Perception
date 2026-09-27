import io
import json
import logging
import queue
import unittest
from collections import deque
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image, ImageDraw

from falcon_perception.paged_inference import PagedInferenceEngine
from falcon_perception.paged_ocr_inference import OCRInferenceEngine
from falcon_perception.server.engine_worker import (
    WorkerRequest,
    _enqueue_layout_request,
    _harvest_compound_crop,
)


class OCRCropGeometryTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (160, 120), "white")
        ImageDraw.Draw(self.image).rectangle((100, 80, 140, 110), fill="blue")
        self.valid = {
            "category": "text",
            "bbox": [100.8, 80.8, 140.6, 110.6],
            "score": 0.9,
        }

    def invalid_boxes(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for index in range(4):
                box = [0.0, 0.0, 150.0, 115.0]
                box[index] = value
                yield box
        yield [10.8, 10.8, 10.6, 30.0]
        yield [10.8, 10.8, 30.0, 10.6]
        yield [10.8, 10.8, 10.8, 30.0]
        yield [10.8, 10.8, 30.0, 10.8]

    def engine(self, box):
        engine = OCRInferenceEngine.__new__(OCRInferenceEngine)
        engine.load_layout_model = Mock()
        engine.run_layout_detection = Mock(
            return_value=[
                [
                    {"category": "text", "bbox": box, "score": 0.9},
                    self.valid,
                ]
            ]
        )
        engine.tokenizer = SimpleNamespace(
            eos_token_id=1,
            end_of_query_token_id=2,
            decode=lambda ids: "valid text",
        )
        engine.waiting = deque()
        engine._compound_state = {}
        return engine

    def assert_valid_crop(self, sequence):
        expected = self.image.crop((100, 80, 141, 111))
        self.assertEqual(sequence._image_pil.size, expected.size)
        self.assertEqual(sequence._image_pil.tobytes(), expected.tobytes())

    def test_generate_with_layout_keeps_valid_regions(self):
        def generate(sequences, **kwargs):
            self.assertEqual(len(sequences), 1)
            self.assert_valid_crop(sequences[0])
            return sequences

        for box in self.invalid_boxes():
            with (
                self.subTest(box=box),
                patch.object(PagedInferenceEngine, "generate", side_effect=generate),
            ):
                results = self.engine(box).generate_with_layout(
                    [self.image], use_tqdm=False
                )
                self.assertEqual(results, [[{**self.valid, "text": "valid text"}]])

    def test_server_assembly_keeps_valid_regions_and_json(self):
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        request = WorkerRequest(
            request_id=7,
            prompt="",
            image_bytes=buffer.getvalue(),
            max_tokens=32,
            min_image_size=64,
            max_image_size=1536,
            task="ocr_layout",
        )
        for box in self.invalid_boxes():
            with self.subTest(box=box):
                engine = self.engine(box)
                responses = queue.Queue()
                _enqueue_layout_request(
                    engine, request, responses, 0, logging.getLogger(__name__)
                )
                self.assertEqual(len(engine.waiting), 1)
                sequence = engine.waiting.popleft()
                self.assert_valid_crop(sequence)
                sequence.input_ids = torch.tensor([1])
                sequence._output_ids = [torch.tensor(3)]
                _harvest_compound_crop(
                    engine, sequence, 7, responses, 0, logging.getLogger(__name__)
                )
                response = responses.get_nowait()
                self.assertIsNone(response.error)
                self.assertEqual(
                    response.result.layout_regions,
                    [{**self.valid, "text": "valid text"}],
                )
                json.dumps(asdict(response), allow_nan=False)
                self.assertEqual(engine._compound_state, {})

    def test_direct_crop_builder_rejects_raw_invalid_boxes_before_rounding(self):
        for box in self.invalid_boxes():
            with self.subTest(box=box):
                results = OCRInferenceEngine.build_crop_sequences(
                    self.image,
                    [{"category": "text", "bbox": box}, self.valid],
                    min_crop_dim=1,
                )
                self.assertEqual([index for _, index in results], [1])
                self.assert_valid_crop(results[0][0])

    def test_valid_crop_rounding_clipping_and_minimum_size_are_preserved(self):
        cases = [
            ([10.2, 10.2, 10.6, 10.6], (10, 10, 11, 11)),
            ([-5.2, -3.2, 10.6, 20.6], (0, 0, 11, 21)),
            ([150.8, 110.8, 180.2, 140.2], (150, 110, 160, 120)),
            ([170.0, 130.0, 180.0, 140.0], None),
        ]
        for box, expected_box in cases:
            with self.subTest(box=box):
                results = OCRInferenceEngine.build_crop_sequences(
                    self.image,
                    [{"category": "text", "bbox": box}],
                    min_crop_dim=1,
                )
                if expected_box is None:
                    self.assertEqual(results, [])
                    continue
                expected = self.image.crop(expected_box)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0][0]._image_pil.size, expected.size)
                self.assertEqual(results[0][0]._image_pil.tobytes(), expected.tobytes())
        self.assertEqual(
            OCRInferenceEngine.build_crop_sequences(
                self.image,
                [{"category": "text", "bbox": cases[0][0]}],
            ),
            [],
        )

    def test_all_invalid_regions_preserve_native_and_server_empty_layout_behavior(self):
        engine = self.engine([0, 0, float("inf"), 120])
        engine.run_layout_detection.return_value = [
            [
                {"category": "text", "bbox": box, "score": 0.9}
                for box in self.invalid_boxes()
            ]
        ]

        def generate(sequences, **kwargs):
            self.assertEqual(len(sequences), 1)
            self.assertEqual(sequences[0]._image_pil.tobytes(), self.image.tobytes())
            return sequences

        with patch.object(PagedInferenceEngine, "generate", side_effect=generate):
            self.assertEqual(
                engine.generate_with_layout([self.image], use_tqdm=False),
                [
                    [
                        {
                            "category": "plain",
                            "bbox": [0, 0, 160, 120],
                            "score": 1.0,
                            "text": "valid text",
                        }
                    ]
                ],
            )
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        request = WorkerRequest(
            7, "", buffer.getvalue(), 32, 64, 1536, task="ocr_layout"
        )
        responses = queue.Queue()
        _enqueue_layout_request(
            engine, request, responses, 0, logging.getLogger(__name__)
        )
        response = responses.get_nowait()
        self.assertIsNone(response.error)
        self.assertEqual(response.result.layout_regions, [])
        self.assertEqual(response.result.text, "")
        self.assertEqual(list(engine.waiting), [])
        self.assertEqual(engine._compound_state, {})


if __name__ == "__main__":
    unittest.main()
