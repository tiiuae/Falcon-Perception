import unittest

from PIL import Image

from falcon_perception.data import resize_image_if_necessary


class ResizeImageIfNecessaryTest(unittest.TestCase):
    def test_panorama_keeps_short_side_at_min_dimension(self):
        # A 10000x100 panorama has an extreme aspect ratio that previously
        # collapsed the short side below min_dimension (e.g. 1024x10).
        img = Image.new("RGB", (10000, 100))
        out = resize_image_if_necessary(img, 256, 1024)
        self.assertGreaterEqual(out.size[0], 256)
        self.assertLessEqual(out.size[0], 1024)
        self.assertGreaterEqual(out.size[1], 256)
        self.assertLessEqual(out.size[1], 1024)

    def test_vertical_panorama_keeps_short_side_at_min_dimension(self):
        img = Image.new("RGB", (100, 10000))
        out = resize_image_if_necessary(img, 256, 1024)
        self.assertGreaterEqual(out.size[0], 256)
        self.assertLessEqual(out.size[0], 1024)
        self.assertGreaterEqual(out.size[1], 256)
        self.assertLessEqual(out.size[1], 1024)

    def test_image_already_in_range_is_unchanged(self):
        img = Image.new("RGB", (512, 512))
        out = resize_image_if_necessary(img, 256, 1024)
        self.assertEqual(out.size, (512, 512))


if __name__ == "__main__":
    unittest.main()
