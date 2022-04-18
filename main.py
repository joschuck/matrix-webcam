"""Shows your webcam video - Matrix style"""

import cv2
import numpy as np

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]
SCALE_X = 6
SCALE_Y = 16


def ascii_image(image: np.ndarray) -> str:
    """Turns a numpy image into rich-CLI ascii image"""
    width, height = image.shape[1], image.shape[0]
    image = cv2.resize(image, (width // SCALE_X, height // SCALE_Y))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ascii_str = ""
    for (y, x), pixel in np.ndenumerate(gray):
        ascii_str += f"{ASCII_CHARS[pixel // 25]}"
        if x == image.shape[1] - 1:
            ascii_str += "\n"
    return ascii_str


def show_matrix():
    cam = cv2.VideoCapture(0)
    while True:
        _, image = cam.read()

        print("\033[H\033[3J", end="")  # clear
        print("\033[92m")  # green color
        print(ascii_image(image))  # ascii image

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_matrix()
