# matrix-webcam

[![License MIT](https://img.shields.io/github/license/joschuck/matrix-webcam.svg)](https://github.com/joschuck/matrix-webcam/blob/main/LICENSE)
[![issues](https://img.shields.io/github/issues/joschuck/matrix-webcam.svg)](https://github.com/joschuck/matrix-webcam/issues)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

This package displays your webcam video feed in the console.

Take your next video conference from within the matrix!

![matrix-webcam demo](https://raw.githubusercontent.com/joschuck/matrix-webcam/main/doc/matrix-webcam02.gif)

## Running it

Make sure you have Python and pip installed. Installation using pip:

    $ python -m pip install matrix-webcam
    $ matrix-webcam

Installing and running it from source:

    $ git clone https://github.com/joschuck/matrix-webcam.git
    $ cd matrix-webcam
    $ python -m pip install .
    $ matrix-webcam

## Can I use this for Teams/Zoom/Skype etc.? 

Yes! You can for example use [OBS Studio](https://obsproject.com/) together with the [Virtual webcam plugin](https://github.com/Fenrirthviti/obs-virtual-cam/releases).
Then all you need to do is select the virtual webcam in Teams/Zoom/Skype.

## Development

I'd recommend creating a new virtual environment (if you are under Ubuntu install it using `sudo apt install python3-venv` using 

    $ python3 -m venv venv/
    $ source venv/bin/activate

Then install the dependencies using:

    $ pip install -e .[dev,deploy]

Setup pre-commit, too:

    $ pre-commit install

### TODO

* [ ] Move to opencv-python-headless
* [ ] add tests
* [ ] add webcam selection
* 

## License
This project is licensed under the MIT License (see the `LICENSE` file for details).
