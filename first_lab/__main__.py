from PIL import Image as PILImage
import numpy as np

from pathlib import Path

from typing import Final
from typing_extensions import Self
from functools import wraps

IMAGES_PATH: Final = Path('./images')

def get_image(image_path: Path) -> np.array:
    image_path = IMAGES_PATH / image_path
    return np.array(PILImage.open(image_path), dtype=np.int32)

def filter(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.image_path = Path('{0}_{1}{2}'.format(
            self.image_path.stem,
            func.__name__,
            self.image_path.suffix,
        ))
        return func(self, *args, **kwargs)
    return wrapper


class Image(object):
    def __init__(self, image_path: Path | str) -> None:
        self.image: np.array = get_image(Path(image_path))
        self.image_path: Path = Path(image_path)

    def save(self) -> None:
        uint8_array = self.image.clip(0, 0xFF).astype(np.uint8)
        PILImage.fromarray(uint8_array).save(
            IMAGES_PATH / self.image_path,
        )
        print('Saved `{0}`'.format(self.image_path))

    @filter
    def inversed(self) -> Self:
        self.image = np.full(self.image.shape, 0xFF, dtype=np.uint8) - self.image
        return self

    @filter
    def rotated_90_clockwise(self) -> Self:
        self.image = np.rot90(self.image, 3)
        return self

    @filter
    def mirrored(self) -> Self:
        height, width, _ = self.image.shape
        half_width = width // 2
        self.image[:, half_width:] = np.flip(self.image[: , :half_width], 1)
        return self

    @filter
    def grayscale(self) -> Self:
        intensity = (
            self.image[:, :, 0] * .36 +
            self.image[:, :, 1] * .53 +
            self.image[:, :, 2] * .11
        )

        self.image[:, :, 0] = intensity
        self.image[:, :, 1] = intensity
        self.image[:, :, 2] = intensity

        return self

    @filter
    def increased_brightness(self, coef: int = 100) -> Self:
        self.image[:, :, 0] += coef
        self.image[:, :, 1] += coef
        self.image[:, :, 2] += coef
        return self


def main():
    image = Image('main_image.jpeg').grayscale()
    image.save()

    image = Image('main_image.jpeg').increased_brightness()
    image.save()

if __name__ == '__main__':
    main()
