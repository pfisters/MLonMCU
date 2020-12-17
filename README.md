<!-- ABOUT THE PROJECT -->
## About The Project

This project is done in the context of the course "Machine Learning on Microcontrollers" @ ETH Zurich.
It contains a reduced Keras implementation of [MTCNN](https://arxiv.org/abs/1604.02878) and is trained using the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset. It also contains a quantization part where the networks (weights and activations) are quantized from float32 to int8 without a major loss of accuarcy.

<!-- INSTALLATION -->
## Installation

1. Clone the repo
2. Setup an environment according to the `environment.yml` file
3. Set the source node as a environment variable
    ```sh
    export PYTHONPATH="path to your directory"
    ```

<!-- USAGE -->
## Usage

1. Download the data with `download_data.py`
   ```sh
   python data/download_data.py
   ```
2. Generate the training images for the pnet, rnet and onet (with arguments 12, 24 and 48 respectively) with `generate_training_data.py`
   ```sh
   python data/generate_training_data.py 12
   ```
3. Train the models with `train_pnet.py`, `train_rnet.py` and `train_onet.py`
   ```sh
   python models/train_pnet.py
   ```
4. Quantize the models with `quantize_pnet.py`, `quantize_rnet.py` and `quantize_onet.py`
    ```sh
    python models/quantize_pnet.py
    ```

In models, you will find `*.h5`, `*.tflite` and `.h` files, for the keras model, tensorflow lite model and a hex representation of them. The quantization files also give you the possibility to generate validation data for STM32 X-CUBE-AI expansion (use the keras models `*.h5` and the `*net_bbx.csv`, `*net_cat.csv` and `*net_data.csv` files). 

To increase the performance, you can also forward sample with `data/generate_hard_samples` for the rnet and onet with the arguments 24 and 48 respectively.

5. To test the performance, run 
  ```sh
  python detect_faces "path to your image" 
  ```
<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
You will recognise some code sections or even entire files from the following repositories:
* [MTCNN](https://github.com/ipazc/mtcnn)
* [MTCNN-Tensorflow](https://github.com/wangbm/MTCNN-Tensorflow)