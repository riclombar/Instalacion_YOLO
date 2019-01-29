## Intro

Tutorial y primeros pasos para instalar YOLO a través de [Darkflow](https://github.com/thtrieu/darkflow), el cual es una herramienta escrita en Python 3 que hace las redes neuronales de código abierto [Darknet](https://pjreddie.com/darknet/) disponibles en Python usando Tensorflow. 

Clasificación y detección de objetos en tiempo real. Artículo: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf).

Se pueden consultar los archivos .weight directo de [este enlace](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU).


## Dependencias

Python3, tensorflow 1.0, numpy, opencv 3.

## Instalación

Seleccionar la carpeta de instalación.
```
cd Documentos
git clone https://github.com/riclombar/Instalacion_YOLO
```
	
Si el comando `git` no se encuentra, se necesita instalar git.
```
sudo apt install git
```
	
Se procede a crear un ambiente virtual dentro de la carpeta clonada de Darkflow. 
```
cd Instalacion_YOLO
virtualenv --python=python3 .venv
source .venv/bin/activate
```
	
Nótese que si se tiene una versión de python 3 más nueva que la versión 3.6, se debe indicar que se cree el ambiente virtual con la versión python 3.6.
```
virtualenv --python=python3.6 .venv
```

Si no se tiene instalado `virtualenv`.
```
sudo apt install virtualenv
```
	
Se necesitan instalar las siguientes librerías.
```
pip install Cython
pip install numpy
pip install tensorflow
pip install opencv-python
pip install .
python setup.py build_ext --inplace
```
	
Para la instalación de tensorflow también puedes ir a la página oficial de [Tensorflow](https://www.tensorflow.org/install/pip) y buscar la versión de Tensorflow correspondiente a tu versión de python, ejemplo para versión de python 3.6 versión de CPU.
```
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
```
	
Con eso termina la instalación de darflow corriendo en Tensorflow con versión de CPU. Para comprobar que darkflow funciona correctamente, se puede usar el siguiente comando:
```
flow --help
```
	
Si se muestra la ayuda del comando flow, todo está bien.

### Instalando componentes de la versión GPU de Tensorflow

Para usar la versión GPU de Tensorflow es necesario instalar los drivers de nvidia, así como el Toolkit de nvidia y cuDNN. Para esto se requieren tarjetas de gráficos habilitados por CUDA. La lista de tarjetas está disponible en: https://developer.nvidia.com/cuda-gpus.

Para instalar los drivers de nvidia es recomendable leer la documentación. Adicionalmente, puedes consultar [esta](http://www.linuxandubuntu.com/home/how-to-install-latest-nvidia-drivers-in-linux) guía.

#### Instalando CUDA Toolkit 9.0

Se procede con la instalación del Toolkit de CUDA. Se necesita descargar el archivo de [aquí](https://developer.nvidia.com/cuda-90-download-archive), es importante que la versión del Toolkit sea la nueve pues Tensoflow está soportado para esa versión. Se procede a seleccionar el sistema operativo, la arquitectura x64, la distribución de linux, la versión de ubuntu (si es ubuntu 18, selecciona ubuntu 17) y, finalmente, el instalador deb. Debe quedar como se muestra [aquí](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=deblocal).

Descarga el primer archivo y los primeros dos parches (hay más parches pero no son necesarios). Para instalar:
```
sudo dpkg -i cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

Para instalar los parches los abrimos con el instalador de software de ubuntu y hacemos la instalación. Necesitas actualizar tu variable PATH.
```
sudo nano ~/.bashrc
```

Ve hasta la última línea y añade las líneas siguientes:
```
export PATH=/usr/local/cuda-9.0/bin${PATH:+:$PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Luego guardas con `CTRL-o`, presionas `ENTER` y cierras con `CTRL-x`.

#### Instalando cuDNN

Ve a https://developer.nvidia.com/cudnn y dirígete a descargas. Se te pide que te registres si es la primera vez, selecciona el link que dice Releases archivados y selecciona la versión 7.0.5 para CUDA Toolkit 9.0 de la fecha 5 de de diciembre de 2017. Descarga la librería para linuz que viene en un archivo `.tar`. Abre una terminal donde guardaste el archivo `.tar` y descomprime usando el comando:
```
tar -xzvf cudnn-9.0-linux-x64-v7.tgz
```

Finalmente, corre los siguientes comandos para mover los archivos correspondientes a tu folder de CUDA.
```
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

## Flowing the graph using `flow`

```bash
# Have a look at its options
flow --h
```

First, let's take a closer look at one of a very useful option `--load`

```bash
# 1. Load tiny-yolo.weights
flow --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights

# 2. To completely initialize a model, leave the --load option
flow --model cfg/yolo-new.cfg

# 3. It is useful to reuse the first identical layers of tiny for `yolo-new`
flow --model cfg/yolo-new.cfg --load bin/tiny-yolo.weights
# this will print out which layers are reused, which are initialized
```

## Training new model

Training is simple as you only have to add option `--train`. Training set and annotation will be parsed if this is the first time a new configuration is trained. To point to training set and annotations, use option `--dataset` and `--annotation`. A few examples:

```bash
# Initialize yolo-new from yolo-tiny, then train the net on 100% GPU:
flow --model cfg/yolo-new.cfg --load bin/tiny-yolo.weights --train --gpu 1.0

# Completely initialize yolo-new and train it with ADAM optimizer
flow --model cfg/yolo-new.cfg --train --trainer adam
```

During training, the script will occasionally save intermediate results into Tensorflow checkpoints, stored in `ckpt/`. To resume to any checkpoint before performing training/testing, use `--load [checkpoint_num]` option, if `checkpoint_num < 0`, `darkflow` will load the most recent save by parsing `ckpt/checkpoint`.

```bash
# Resume the most recent checkpoint for training
flow --train --model cfg/yolo-new.cfg --load -1

# Test with checkpoint at step 1500
flow --model cfg/yolo-new.cfg --load 1500

# Fine tuning yolo-tiny from the original one
flow --train --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights
```

### Training on your own dataset

*The steps below assume we want to use tiny YOLO and our dataset has 3 classes*

1. Create a copy of the configuration file `tiny-yolo-voc.cfg` and rename it according to your preference `tiny-yolo-voc-3c.cfg` (It is crucial that you leave the original `tiny-yolo-voc.cfg` file unchanged, see below for explanation).

2. In `tiny-yolo-voc-3c.cfg`, change classes in the [region] layer (the last layer) to the number of classes you are going to train for. In our case, classes are set to 3.
    
    ```python
    ...

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    bias_match=1
    classes=3
    coords=4
    num=5
    softmax=1
    
    ...
    ```

3. In `tiny-yolo-voc-3c.cfg`, change filters in the [convolutional] layer (the second to last layer) to num * (classes + 5). In our case, num is 5 and classes are 3 so 5 * (3 + 5) = 40 therefore filters are set to 40.
    
    ```python
    ...

    [convolutional]
    size=1
    stride=1
    pad=1
    filters=40
    activation=linear

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    
    ...
    ```

4. Change `labels.txt` to include the label(s) you want to train on (number of labels should be the same as the number of classes you set in `tiny-yolo-voc-3c.cfg` file). In our case, `labels.txt` will contain 3 labels.

    ```
    label1
    label2
    label3
    ```
5. Reference the `tiny-yolo-voc-3c.cfg` model when you train.

    `flow --model cfg/tiny-yolo-voc-3c.cfg --load bin/tiny-yolo-voc.weights --train --annotation train/Annotations --dataset train/Images`


* Why should I leave the original `tiny-yolo-voc.cfg` file unchanged?
    
    When darkflow sees you are loading `tiny-yolo-voc.weights` it will look for `tiny-yolo-voc.cfg` in your cfg/ folder and compare that configuration file to the new one you have set with `--model cfg/tiny-yolo-voc-3c.cfg`. In this case, every layer will have the same exact number of weights except for the last two, so it will load the weights into all layers up to the last two because they now contain different number of weights.


## Camera/video file demo

For a demo that entirely runs on the CPU:

```bash
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi
```

For a demo that runs 100% on the GPU:

```bash
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi --gpu 1.0
```

To use your webcam/camera, simply replace `videofile.avi` with keyword `camera`.

To save a video with predicted bounding box, add `--saveVideo` option.


## Créditos

Los créditos de este código son de https://github.com/thtrieu. Así como el tutorial de instalación en [este gist](https://gist.githubusercontent.com/simonw/0f93bec220be9cf8250533b603bf6dba/raw/6ce2fb8be577abe8f94adbcab18fd54fd29f93d1/darkflow-osx.md)
