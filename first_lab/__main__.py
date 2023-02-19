from functools import wraps
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image as PILImage
from typing_extensions import Self

IMAGES_PATH: Final = Path('./images')


def get_image(image_path: Path) -> np.array:
    image_path = IMAGES_PATH / image_path
    return np.array(PILImage.open(image_path), dtype=int)


def filter(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.image_path = Path('{0}_{1}{2}'.format(
            self.image_path.stem,
            func.__name__,
            self.image_path.suffix,
        ))
        result = func(self, *args, **kwargs)
        self.image = self.image.clip(0, 0xFF)
        return result
    return wrapper


class Image(object):
    def __init__(self, image_path: Path | str) -> None:
        self.image: np.array = get_image(Path(image_path))
        self.image_path: Path = Path(image_path)

    def save(self) -> None:
        uint8_array = self.image.astype(np.uint8)
        PILImage.fromarray(uint8_array).save(
            IMAGES_PATH / self.image_path,
        )
        print('Saved `{0}`'.format(self.image_path))

    @filter
    def inversed(self) -> Self:
        self.image = np.full(self.image.shape, 0xFF) - self.image
        return self

    @filter
    def rotated_90_clockwise(self) -> Self:
        self.image = np.rot90(self.image, 3)
        return self

    @filter
    def mirrored(self) -> Self:
        _, width, _ = self.image.shape
        half_width = width // 2
        self.image[:, half_width + width % 2:] = np.flip(self.image[:, :half_width], 1)
        return self

    @filter
    def grayscale(self) -> Self:
        intensity = (
            self.image[:, :, 0] * .36 +
            self.image[:, :, 1] * .53 +
            self.image[:, :, 2] * .11
        )

        for k in range(3):
            self.image[:, :, k] = intensity

        return self

    @filter
    def increased_brightness(self, coef: int = 100) -> Self:
        for k in range(3):
            self.image[:, :, k] += coef
        return self

    @filter
    def blur(self, radius: int = 20) -> Self:
        max_sum = (radius*2+1) ** 2 * 0xFF
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                for k in range(3):
                    self.image[i, j, k] = np.sum(self.image[
                        np.maximum(i-radius, 0):np.minimum(i+radius+1, self.image.shape[0]),
                        np.maximum(j-radius, 0):np.minimum(j+radius+1, self.image.shape[1]),
                        k,
                    ]) / max_sum * 0xFF
        return self

    @filter
    def compressed(self, scale: int = 2):
        height, width, channels = self.image.shape
        new_height = int(np.ceil(height / scale))
        new_width = int(np.ceil(width / scale))
        compressed_image = np.empty(
            (new_height, new_width, channels),
            dtype=int,
        )
        for i in range(0, height, scale):
            for j in range(0, width, scale):
                for k in range(3):
                    compressed_image[i // scale, j // scale, k] = np.average(
                        self.image[i:i+scale, j:j+scale, k],
                    )
        self.image = compressed_image
        return self


def main():
    image = Image('main_image.jpeg').mirrored()
    image.save()

    image = Image('main_image.jpeg').blur(10)
    image.save()


if __name__ == '__main__':
    main()
