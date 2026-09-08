import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from falcon_perception.paged_ocr_inference import (
    OCRInferenceEngine,
    PagedInferenceEngine,
)


class LayoutTest(unittest.TestCase):
    def test_layout_preserves_images_and_overlapping_text(self):
        detections = [
            {"category": "picture", "bbox": [0, 0, 100, 100], "score": 0.95},
            {"category": "text", "bbox": [0, 0, 100, 100], "score": 0.9},
            {"category": "caption", "bbox": [10, 10, 80, 40], "score": 0.85},
        ]
        engine = object.__new__(OCRInferenceEngine)
        engine.tokenizer = SimpleNamespace(end_of_query_token_id=1, eos_token_id=2)
        engine.load_layout_model = lambda *args: None
        engine.run_layout_detection = lambda images, **kwargs: [
            detections for _ in images
        ]
        engine._decode_seq_text = lambda seq: "recognized"
        with patch.object(
            PagedInferenceEngine, "generate", lambda self, seqs, **kwargs: seqs
        ):
            results = engine.generate_with_layout([Image.new("RGB", (100, 100))])
        self.assertEqual(
            [r["category"] for r in results[0]], ["picture", "text", "caption"]
        )
        self.assertEqual(
            [r["text"] for r in results[0]], ["", "recognized", "recognized"]
        )


class EvidenceTest(unittest.TestCase):
    def test_native_probabilities_and_termination(self):
        import torch
        from falcon_perception.paged_inference import Sequence
        from falcon_perception.paged_ocr_inference import dedup_overlapping_detections

        engine = object.__new__(OCRInferenceEngine)
        engine.tokenizer = SimpleNamespace(end_of_query_token_id=1, eos_token_id=2)
        engine._decode_seq_text = lambda seq: "hello"

        def generate(self, seqs, **kwargs):
            for seq in seqs:
                seq._output_ids = [torch.tensor(9), torch.tensor(1)]
                seq._output_probs = [torch.tensor(0.8), torch.tensor(0.9)]
            return seqs

        with patch.object(PagedInferenceEngine, "generate", generate):
            image = Image.new("RGB", (100, 100))
            self.assertEqual(engine.generate_plain([image]), ["hello"])
            result = engine.generate_plain([image], return_generation=True)[0]
            self.assertEqual(result["generation"]["token_ids"], [9, 1])
            self.assertAlmostEqual(result["generation"]["token_probabilities"][0], 0.8)
            self.assertTrue(result["generation"]["stop_token_seen"])
        seq = Sequence(text="", image=None)
        self.assertFalse(engine._generation_details(seq, 0.0)["stop_token_seen"])
        seq._output_ids = [torch.tensor(9)]
        seq._output_probs = [torch.tensor(0.8)]
        self.assertFalse(engine._generation_details(seq, 0.0)["stop_token_seen"])
        dets = [
            {"category": "text", "bbox": [0, 0, 100, 100], "score": score}
            for score in (0.7, 0.9)
        ]
        self.assertEqual(dedup_overlapping_detections(dets), [dets[1]])

    def test_image_fallback_retains_box_and_generation(self):
        import torch

        det = {"category": "image", "bbox": [0, 0, 100, 100], "score": 0.9}
        engine = object.__new__(OCRInferenceEngine)
        engine.tokenizer = SimpleNamespace(end_of_query_token_id=1, eos_token_id=2)
        engine.load_layout_model = lambda *args: None
        engine.run_layout_detection = lambda *args, **kwargs: [[det]]
        engine._decode_seq_text = lambda seq: "hello"

        def generate(self, sequences, **kwargs):
            sequences[0]._output_ids = [torch.tensor(1)]
            sequences[0]._output_probs = [torch.tensor(0.9)]
            return sequences

        with patch.object(PagedInferenceEngine, "generate", generate):
            result = engine.generate_with_layout(
                [Image.new("RGB", (100, 100))], return_generation=True
            )[0]
        self.assertEqual([r["category"] for r in result], ["image", "plain"])
        self.assertNotIn("generation", result[0])
        self.assertTrue(result[1]["generation"]["stop_token_seen"])

    def test_server_picture_only_response(self):
        import io
        import logging
        from queue import Queue
        from falcon_perception.server.engine_worker import (
            WorkerRequest,
            _enqueue_layout_request,
        )

        dets = [{"category": "picture", "bbox": [0, 0, 100, 100], "score": 0.9}]
        engine = SimpleNamespace(
            load_layout_model=lambda: None,
            run_layout_detection=lambda *args, **kwargs: [dets],
            tokenizer=SimpleNamespace(eos_token_id=1, end_of_query_token_id=2),
        )
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buffer, format="PNG")
        req = WorkerRequest(1, "", buffer.getvalue(), 100, 64, 1024)
        queue = Queue()
        _enqueue_layout_request(engine, req, queue, 0, logging.getLogger())
        self.assertEqual(
            queue.get_nowait().result.layout_regions, [{**dets[0], "text": ""}]
        )


class WorkerFlowTest(unittest.TestCase):
    def test_server_mixed_regions(self):
        import io
        import logging
        import torch
        from collections import deque
        from queue import Queue
        from falcon_perception.server.engine_worker import (
            WorkerRequest,
            _enqueue_layout_request,
            _harvest_done,
        )

        dets = [
            {"category": "picture", "bbox": [0, 0, 100, 100], "score": 0.95},
            {"category": "text", "bbox": [0, 0, 100, 100], "score": 0.9},
        ]
        engine = SimpleNamespace(
            load_layout_model=lambda: None,
            run_layout_detection=lambda *args, **kwargs: [dets],
            tokenizer=SimpleNamespace(
                eos_token_id=1, end_of_query_token_id=2, decode=lambda ids: "hello"
            ),
            waiting=deque(),
            done=deque(),
            _compound_state={},
        )
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buffer, format="PNG")
        req = WorkerRequest(1, "", buffer.getvalue(), 100, 64, 1024)
        queue = Queue()
        _enqueue_layout_request(engine, req, queue, 0, logging.getLogger())
        self.assertEqual(len(engine.waiting), 1)
        seq = engine.waiting.popleft()
        seq.input_ids = torch.tensor([3])
        seq._output_ids = [torch.tensor(4)]
        engine.done.append(seq)
        _harvest_done(engine, queue, 0, logging.getLogger())
        result = queue.get_nowait().result
        self.assertEqual(
            [r["category"] for r in result.layout_regions], ["picture", "text"]
        )
        self.assertEqual(result.text, "hello")
        self.assertFalse(engine._compound_state)


if __name__ == "__main__":
    unittest.main()
