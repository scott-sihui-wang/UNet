# UNet

## 1. Introduction

In this project, I re-implemented `UNet` and tested its segmentation performance. I did a bunch of experiments to see how different model parameters and training parameters can affect `UNet`'s performance on segmentation. 

Please refer to my [report](report.pdf) to see full details, i.e., technical discussions, how to run the code, qualitative and quantitative results.

### Experiments

#### Tweaking Training Parameters

- **Default** (**Training Epochs:** 50, **Batch Size:** 4, **Learning Rate:** 0.01) [Colab Page](https://colab.research.google.com/drive/1OrMHf6vgV9qUkFfyK-rsCIcnqyQ07zgQ?usp=sharing)

- **Data Augmentation** (with **no** augmentation) [Colab Page](https://colab.research.google.com/drive/17rh0JIR1UoA6wgAwijWVzrRxri1l6_2Z?usp=sharing)

- **Learning Rate** 0.1 [Colab Page](https://colab.research.google.com/drive/1CXYFQuHd_fYep--I9cHP8k_uRbH_9TKW?usp=sharing) 0.05 [Colab Page](https://colab.research.google.com/drive/1TxWd0FIXm9WFrligMRRt71kmShjJrEyB?usp=sharing) 0.02 [Colab Page](https://colab.research.google.com/drive/1ufQC_MtHNtuCsMA-2j2geuGVoutC04_c?usp=sharing) 0.005[Colab Page](https://colab.research.google.com/drive/18jzpLEoA2l1WiUcPuQ9iuOFDYIUsBglv?usp=sharing)

- **Batch Size** 1 [Colab Page](https://colab.research.google.com/drive/1l0bqMQ2u9rKUokUxYp3wT3o57Syrc9q7?usp=sharing) 2 [Colab Page](https://colab.research.google.com/drive/13hwQEOAGq2qawGWVlHq96b-gfmYcG36l?usp=sharing) 8 [Colab Page](https://colab.research.google.com/drive/1PynF7w9my3_RIOa0xioAGd7Vk0Cxdh64?usp=sharing)

- **Input Image Size** 316 $\times$ 316 [Colab Page](https://colab.research.google.com/drive/1UIBThWumOH8IvCT6AJE4uxNW6okk0niy?usp=sharing) 700 $\times$ 700 [Colab Page](https://colab.research.google.com/drive/1zKMOXB9JNb3YV2pCH8Y_usvRaUbeoz9W?usp=sharing)

- **Padding** (**padding**=1) [Colab Page](https://colab.research.google.com/drive/1r02ngXNLQD2xrzwtuhM5gSdUuEudPFkU?usp=sharing)

####  Tweaking Model Structures

- **Depth of Network** 6 Layers [Colab Page](https://colab.research.google.com/drive/13QDhcDey4AgkcfUhJ8n7uxUPeBQZiAPr?usp=sharing) 4 Layers [Colab Page](https://colab.research.google.com/drive/11fb6kZbY2ZnUIANqaK0f2Gvr7sVy37A7?usp=sharing)

- **Number of Convolutional Kernels** double the number of channels [Colab Page](https://colab.research.google.com/drive/1y5wm7He3_VPRLRM4KjYeZU6Jc5zH20YE?usp=sharing) halve the number of channels [Colab Page](https://colab.research.google.com/drive/1BUmknR45V89d8oy_YjSxEK0j9Bgc7khG?usp=sharing)

**Topics:** _Computer Vision_, _UNet_, _Medical Image Processing_, _Semantic Segmentation_

**Skills:** _Pytorch_, _Python_, _Deep Neural Networks_, _Jupyter Lab_, _Colab_

## 2. Demo
