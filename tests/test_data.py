import tempfile
import unittest

import numpy as np
from PIL import Image

from falcon_perception.data import load_image


class LoadImageNpyTest(unittest.TestCase):
    def test_load_image_from_npy(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".npy") as f:
            np.save(f.name, arr)
            img = load_image(f.name)

        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (32, 32))
        self.assertEqual(img.mode, "RGB")

    def test_load_image_from_npy_grayscale(self):
        arr = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".npy") as f:
            np.save(f.name, arr)
            img = load_image(f.name)

        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (16, 16))


if __name__ == "__main__":
    unittest.main()
