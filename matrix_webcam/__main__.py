"""Shows your webcam video - Matrix style"""
import argparse
import os
import cv2
import numpy as np

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def ascii_image(image: np.ndarray, width: int, height: int) -> str:
    """Turns a numpy image into rich-CLI ascii image"""
    image = cv2.resize(image, (width, height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ascii_str = ""
    for (_, x), pixel in np.ndenumerate(gray):  # pylint: disable=C0103
        ascii_str += f"{ASCII_CHARS[pixel // 25]}"
        if x == image.shape[1] - 1:
            ascii_str += "\n"
    return ascii_str


def parse_args() -> argparse.Namespace:
    """Parses width and height in characters from CLI."""
    parser = argparse.ArgumentParser(description="py-darts")
    parser.add_argument(
        "width",
        nargs="?",
        default=80,
        type=int,
        help="Width of the video in characters.",
    )

    parser.add_argument(
        "height",
        nargs="?",
        default=23,
        type=int,
        help="Height of the video in characters.",
    )
    return parser.parse_args()


def main() -> None:
    """Main loop."""
    args = parse_args()
    os.system("cls" if os.name == "nt" else "clear")
    cam = cv2.VideoCapture(0)
    try:
        while True:
            _, image = cam.read()

            print("\033[H\033[3J", end="")  # clear
            print("\033[92m")  # green color
            print(ascii_image(image, args.width, args.height))  # ascii image
    except KeyboardInterrupt:
        print("Shutdown")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
