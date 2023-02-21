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
    def blur_faster(self, radius: int = 20) -> Self:
        max_sum = (radius*2+1) ** 2 * 0xFF
        partialSum = np.empty(self.image.shape)
        partialSum[0,0,:] = self.image[0,0,:]
        for i in range(1, partialSum.shape[0]):
            partialSum[i, 0, :] = partialSum[i-1, 0, :] + self.image[i, 0, :]
        for i in range(1, partialSum.shape[1]):
            partialSum[0, i, :] = partialSum[0, i-1, :] + self.image[0, i, :]
        for i in range(1, partialSum.shape[0]):
            for j in range(1, partialSum.shape[1]):
                for k in range(partialSum.shape[2]):
                    partialSum[i,j,k] = partialSum[i-1,j,k] + partialSum[i,j-1,k] + self.image[i,j,k] - partialSum[i-1,j-1,k]
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                    bbound = np.minimum(i + radius, self.image.shape[0]-1)
                    rbound = np.minimum(j + radius, self.image.shape[1]-1)
                    self.image[i,j,:] = partialSum[bbound, rbound, :]
                    if i > radius:
                        self.image[i,j,:] = self.image[i,j,:] - partialSum[i-radius, rbound, :]
                    if j > radius:
                        self.image[i,j,:] = self.image[i,j,:] - partialSum[bbound, j-radius, :]
                    if i > radius and j > radius:
                        self.image[i,j,:] = self.image[i,j,:] + partialSum[i-radius, j-radius, :]

                    self.image[i,j,:] = self.image[i,j,:] / max_sum * 0xFF
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

    @filter
    def sepia(self) -> Self:
        sepia_matrix = (
            (0.393, 0.769, 0.189),
            (0.349, 0.686, 0.168),
            (0.272, 0.534, 0.131),
        )
        for k in range(3):
            r, g, b = sepia_matrix[k]
            self.image[:, :, k] = (
                r * self.image[:, :, 0] +
                g * self.image[:, :, 1] +
                b * self.image[:, :, 2]
            )

        return self

    @filter
    def average_color(self, coefficient: float = 0.5):
        mid_rgb_color = (
            np.average(self.image[:, :, 0]),
            np.average(self.image[:, :, 1]),
            np.average(self.image[:, :, 2]),
        )
        height, width, _ = self.image.shape
        print(mid_rgb_color)
        for i in range(height):
            for j in range(width):
                self.image[i, j] = [x + (y - x) * coefficient for x, y in zip(self.image[i, j], mid_rgb_color)]
        return self

def main():
    image_path = 'pasha.jpg'
    image = Image(image_path).mirrored()
    image.save()

    image = Image(image_path).sepia()
    image.save()

    image = Image(image_path).average_color()
    image.save()


if __name__ == '__main__':
    main()
