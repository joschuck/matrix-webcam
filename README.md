# matrix-webcam

[![License MIT](https://img.shields.io/github/license/joschuck/matrix-webcam.svg)](https://github.com/joschuck/matrix-webcam/blob/main/LICENSE)
[![issues](https://img.shields.io/github/issues/joschuck/matrix-webcam.svg)](https://github.com/joschuck/matrix-webcam/issues)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

This package displays your webcam video feed in the console.

Take your next video conference from within the matrix!

![matrix-webcam demo](https://raw.githubusercontent.com/joschuck/matrix-webcam/main/doc/matrix-webcam.gif)

## Running it

Make sure you have Python and pip installed. Installation using pip:

    $ python -m pip install matrix-webcam
    $ matrix-webcam

Installing and running it from source:

    $ git clone https://github.com/joschuck/matrix-webcam.git
    $ cd matrix-webcam
    $ python -m pip install .
    $ matrix-webcam


## Can I change the size or resolution

Yes! Just add the desired width and height in characters to the command, i.e.

    $ python main.py 120 30

## Can I use this for Teams/Zoom/Skype etc.? 

Yes! You can for example use [OBS Studio](https://obsproject.com/) together with the [Virtual webcam plugin](https://github.com/Fenrirthviti/obs-virtual-cam/releases).
Then all you need to do is select the virtual webcam in Teams/Zoom/Skype.

## License
This project is licensed under the MIT License (see the `LICENSE` file for details).
