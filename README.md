# minDL
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Training XOR neural network](./media/xor.gif)

Minimalistic set of tools for Deep Learning.
The project is used as part of video course on YouTube.
([English](https://www.youtube.com/playlist?list=PL0SnwWLf9Xb6fmT4GspEB_Ek4E6oiSDi-)|[Russian](https://www.youtube.com/playlist?list=PL0SnwWLf9Xb6ScpdDuveGNGMUUpSvwACz))

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

## Neural network implementations
There are 2 implementations of a feedforward neural networks so far: sequential and vectorized.

`plaindl` - Implements the training algorithm on pure python.
The purpose of this realisation is just to show, that there's no "magic" in the neural networks.

`mindl` - Implements the training algorithm using only `numpy` for matrix computations and code simplification.

### Train your own network
You could either run the `run.py` script or start JupyterLab and check out more examples:
```shell
jupyter lab
```
If the Lab is not opened automatically, then open http://localhost:8888/lab
