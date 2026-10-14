import io
import unittest

import numpy as np
from PIL import Image

from falcon_perception.data import load_image


def _png_bytes(size=(8, 8), color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class LoadImageTest(unittest.TestCase):
    def test_accepts_plain_bytes(self):
        raw = _png_bytes()
        img = load_image(raw)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (8, 8))

    def test_accepts_numpy_bytes_scalar(self):
        raw = _png_bytes()
        img = load_image(np.bytes_(raw))
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (8, 8))

    def test_accepts_ndarray(self):
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        img = load_image(arr)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (8, 8))


if __name__ == "__main__":
    unittest.main()
