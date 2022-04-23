"""Shows your webcam video - Matrix style"""
import argparse
import os
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from mediapipe.python.solutions.selfie_segmentation import SelfieSegmentation

ASCII_CHARS = [" ", "@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def ascii_image(image: npt.NDArray[np.uint8], width: int, height: int) -> str:
    """Turns a numpy image into rich-CLI ascii image"""
    image = cv2.resize(image, (width, height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ascii_str = ""
    for (_, x), pixel in np.ndenumerate(gray):  # pylint: disable=C0103
        ascii_str += f"{ASCII_CHARS[int(pixel / (256 / len(ASCII_CHARS)))]}"
        if x == image.shape[1] - 1:
            ascii_str += "\n"
    return ascii_str


def parse_args() -> argparse.Namespace:
    """Parses width and height in characters from CLI."""
    parser = argparse.ArgumentParser(description="py-darts")
    parser.add_argument(
        "width",
        nargs="?",
        default=120,
        type=int,
        help="Width of the video in characters.",
    )

    parser.add_argument(
        "height",
        nargs="?",
        default=30,
        type=int,
        help="Height of the video in characters.",
    )
    return parser.parse_args()


def main() -> None:
    """Main loop."""
    args = parse_args()
    os.system("cls" if os.name == "nt" else "clear")
    cap = cv2.VideoCapture(0)
    bg_image: Optional[npt.NDArray[np.uint8]] = None
    with SelfieSegmentation(model_selection=1) as selfie_segmentation:
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = selfie_segmentation.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.9
            # The background can be customized.
            #   a) Load an image (with the same width and height of the input image) to
            #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
            #   b) Blur the input image by applying image filtering, e.g.,
            #      bg_image = cv2.GaussianBlur(image,(55,55),0)
            if bg_image is None:
                bg_image = np.zeros(image.shape, dtype=np.uint8)
            output_image = np.where(condition, image, bg_image)

            print("\033[H\033[3J", end="")  # clear
            print("\033[92m")  # green color
            print(ascii_image(output_image, args.width, args.height))  # ascii image
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
