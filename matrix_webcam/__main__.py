"""Shows your webcam video - Matrix style"""
import argparse
import random
import signal
import time
from string import printable
from typing import Optional, Any
from itertools import chain

import _curses
import cv2
import numpy as np
import numpy.typing as npt
from mediapipe.python.solutions.selfie_segmentation import SelfieSegmentation

curses: Any = _curses  # ignore mypy
ASCII_CHARS = np.array([" ", "@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."])


def ascii_image2(
    image: npt.NDArray[np.uint8], width: int, height: int, linebreak: bool = False
) -> str:
    """Turns a numpy image into rich-CLI ascii image"""
    gray = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    gray = (gray / (256 / len(ASCII_CHARS))).astype(int)
    gray = np.ascontiguousarray(np.take(ASCII_CHARS, gray))
    return gray.ravel()[:-1]
    

def parse_args() -> argparse.Namespace:
    """Parses width and height in characters from CLI."""
    parser = argparse.ArgumentParser(description="matrix-webcam")
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        help="Sets the index of the webcam if you have more than one webcam.",
    )
    parser.add_argument(
        "-l",
        "--letters",
        type=int,
        default=2,
        help="The number of letters produced per update.",
    )
    parser.add_argument(
        "-p",
        "--probability",
        type=int,
        default=5,
        help="1/p probability of a dispense point deactivating each tick.",
    )
    parser.add_argument(
        "-u",
        "--updates-per-second",
        type=int,
        default=15,
        help="The number of updates to perform per second.",
    )
    return parser.parse_args()


def main() -> None:
    """Main loop."""
    args = parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print("No VideoCapture found!")
        cap.release()
        return

    # os.system("cls" if os.name == "nt" else "clear")
    stdscr = init_curses()

    signal.signal(signal.SIGINT, lambda signal, frame: terminate(cap, stdscr))

    size = stdscr.getmaxyx()
    height, width = size

    # background is a matrix of the actual letters (not lit up) -- the underlay.
    # foreground is a binary matrix representing the position of lit letters -- the overlay.
    # dispense is where new 'streams' of lit letters appear from.
    printable_chars = printable.strip()
    background = rand_string(printable_chars, width * height)
    foreground: list[tuple[int, int]] = []
    dispense: list[int] = []

    delta = 0
    bg_refresh_counter = random.randint(3, 7)
    perf_counter = time.perf_counter()
    frame_counter = 0
    time_counter = time.monotonic_ns()

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
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.95
            # The background can be customized.
            #   a) Load an image (with the same width and height of the input image) to
            #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
            #   b) Blur the input image by applying image filtering, e.g.,
            #      bg_image = cv2.GaussianBlur(image,(55,55),0)
            if bg_image is None:
                bg_image = np.zeros(image.shape, dtype=np.uint8)
            output_image = np.where(condition, image, bg_image)

            stdscr.clear()

            string = ascii_image2(output_image, width, height)
            for idx, val in enumerate(string):
                stdscr.addstr(idx // width, idx % width, val, curses.color_pair(1))

            now = time.perf_counter()
            delta += (now - perf_counter) * abs(args.updates_per_second)
            perf_counter = now
            update_matrix = delta >= 1

            foreground2 = []
            for idx, (row, col) in enumerate(foreground):
                if row < size[0] - 1:
                    stdscr.addstr(
                        row,
                        col,
                        background[row * size[0] + col],
                        curses.color_pair(1),
                    )

                    if update_matrix:
                        foreground2.append((row + 1, col))
                    else:
                        foreground2.append((row, col))
            foreground = foreground2

            if update_matrix:
                dispense_new = random.choices(range(0, width), k=abs(args.letters))
                dispense2 = []
                for idx, column in enumerate(chain(dispense, dispense_new)):
                    foreground.append((0, column))
                    if random.randint(0, args.probability - 1):
                        dispense2.append(column)
                dispense = dispense2
                delta -= 1

            bg_refresh_counter -= 1
            if bg_refresh_counter <= 0:
                background = rand_string(printable_chars, height * width)
                bg_refresh_counter = random.randint(3, 7)

            stdscr.refresh()
            frame_counter += 1

            # compute fps
            curr = time.monotonic_ns()
            if curr - time_counter > 1e9:
                fps = frame_counter / (curr - time_counter) * 1e9
                stdscr.addstr(0, 0, f"FPS: {fps}")
                time_counter = curr
                frame_counter = 0

            stdscr.nodelay(True)  # Don't block waiting for input.
            char_input = stdscr.getch()
            if cv2.waitKey(1) & 0xFF == 27 or char_input in (3, 27):  # ESC pressed
                break

    terminate(cap, stdscr)


def terminate(cap: Any, stdscr: Any) -> None:
    """# OpenCV and curses shutdown"""
    cap.release()
    cv2.destroyAllWindows()

    curses.curs_set(True)
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()


def init_curses() -> Any:
    """Initializes curses library"""
    stdscr = curses.initscr()
    curses.curs_set(False)  # no blinking cursor
    stdscr.keypad(True)  # if not set will end program on arrow keys etc
    curses.noecho()  # do not echo keypress

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    return stdscr


def rand_string(character_set: str, length: int) -> list:
    """
    Returns a random string.
    character_set -- the characters to choose from.
    length        -- the length of the string.
    """
    return random.choices(character_set, k=length)


if __name__ == "__main__":
    main()
