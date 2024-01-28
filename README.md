# minDL
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Minimalistic set of tools for Deep Learning.

**!NB, this repository is for educational purposes and is not intended for production use.
For production use PyTorch, Tensorflow, etc.**

## Getting started
1. Create a new environment
    ```shell
    python -m venv venv
    ```
2. Activate the new environment (for Windows the activate script will have a different name)
    ```shell
    source venv/bin/activate
    ```
3. Install dependencies
    ```shell
    pip install -r requirements.txt
    ```

## Neural network
There are 2 implementations of a feedforward neural networks so far: sequential and vectorized.

`plaindl` - Implements the training algorithm on pure python.
The purpose of this realisation is just to show, that there's no "magic" in the neural networks.

`mindl` - Implements the training algorithm using only `numpy` for matrix computations and code simplification.
