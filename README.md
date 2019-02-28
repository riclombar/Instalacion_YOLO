## Intro

Tutorial y primeros pasos para instalar YOLO a través de [Darkflow](https://github.com/thtrieu/darkflow), el cual es una herramienta escrita en Python 3 que hace las redes neuronales de código abierto [Darknet](https://pjreddie.com/darknet/) disponibles en Python usando Tensorflow. 

Clasificación y detección de objetos en tiempo real. Artículo: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf).

Se pueden consultar los archivos .weight directo de [este enlace](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU).

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

Ve a https://developer.nvidia.com/cudnn y dirígete a descargas. Se te pide que te registres si es la primera vez, selecciona el link que dice Releases archivados y selecciona la versión 7.0.5 para CUDA Toolkit 9.0 de la fecha 5 de de diciembre de 2017. Descarga la librería para linux que viene en un archivo `.tar`. Abre una terminal donde guardaste el archivo `.tar` y descomprime usando el comando:
```
tar -xzvf cudnn-9.0-linux-x64-v7.tgz
```

Finalmente, corre los siguientes comandos para mover los archivos correspondientes a tu folder de CUDA.
```
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

#### Instalando Tensorflow GPU

En la [página oficial de Tensorflow](https://www.tensorflow.org/install/pip) consulta  la versión que quieres instalar y corre el comando:
```
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl
```

Donde la URL corresponde con la versión que elegiste.

## Abre tu ambiente virtual

Párate sobre la carpeta donde clonaste el respositorio
```
cd Instalacion_YOLO
```

Con el siguiente comando abres tu ambiente virtual
```
source .venv/bin/activate
```

## Pruebas iniciales

Primero necesitas descargar los weights de YOLO y ponerlos en una carpeta bin. Los weights los puedes encontrar en la página de Darknet de Joseph Redmon. El problema es que los actualiza y darkflow funciona con las versiones iniciales de los weights, por lo tanto, los puedes descargar de este [link](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU).

Una vez que descargaste un weight, lo pones en una carpeta que nombraremos bin.
```
mkdir bin
```

Con este comando usas las fotos que están disponibles en la carpeta sample para probar los weights que descargaste.
```
flow --imgdir sample_img/ --model cfg/yolo.cfg --load bin/yolo.weights
```

Si tienes fotos que quieras probar crea una carpeta con tus fotos y corre:
```
flow --imgdir my_photos/ --model cfg/yolo.cfg --load bin/yolo.weights
open my_photos/out/
```

Los resultados se muestran en una carpeta llamada 'out' dentro del directorio de las fotos.

Adicionalmente, puedes probar el algoritmo con la cámara de tu ordenador o con un video.
Con video:
```
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi
```

Si queres que tu modelo corra con GPU solo agrega '--gpu 1.0' al final.

```bash
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi --gpu 1.0
```

Para usar la webcam, solo reemplaza `videofile.avi` con la palabra `camera`.
Para guardar un video añade `--saveVideo` al final.

## Entrenando un modelo nuevo

Para realizar el entrenamiento solo debes añadir la opción `--train`.
```
# Inicializa yolo-new desde yolo-tiny, luego entrena la red con 100% GPU:
flow --model cfg/yolo-new.cfg --load bin/tiny-yolo.weights --train --gpu 1.0
```
Durante el entrenamiento, el código guarda ocasionalmente los resultados en checkpoints de tensorflow, guardados en `ckpt/`. Para continuar a partir de cualquier checkpoint antes de reanudar el entrenamiento, usa `--load (número de checkpoint)`, si `checkpoint_num < 0`, `darkflow` cargará el checkpoint más reciente.

```
# Continuar desde el checkpoint más reciente
flow --train --model cfg/yolo-new.cfg --load -1

# Continuar con el checkpoint en el paso 1500
flow --model cfg/yolo-new.cfg --load 1500

# Afinar yolo-tiny a partir del original
flow --train --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights
```

### Entrenar con tu propio dataset

*Los pasos siguientes asumen que queremos usar tiny YOLO y nuestro dataset tiene 3 clases*

1. Crea una copia del archivo de configuración `tiny-yolo-voc.cfg` y renómbrala de acuerdo con tu preferencia `tiny-yolo-voc-3c.cfg` (es crucial que dejes el archivo original sin cambios).

2. En `tiny-yolo-voc-3c.cfg`, cambia las clases en la capa [region] (la última) para el número de clases que vas a entrenar. En este caso, classes se pone en 3. 
    ```
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

3. En `tiny-yolo-voc-3c.cfg`, cambia los filtros en la capa [convolutional] (penúltima) por num * (classes + 5). En nuestro caso, num es 5 y classes son 3 entonces 5 * (3 + 5) = 40 por tanto, filters se pone en 40.   
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

4. Cambia `labels.txt` para incluir las etiquetas con las que quieras entrenar (el número de etiquetas debe ser el mismo que el número de clases que pusiste en `tiny-yolo-voc-3c.cfg`). En nuestro caso, `labels.txt` contendrá tres etiquetas.
    ```
    persona
    pasillo
    puerta
    ```
    
5. Haz referencia al modelo `tiny-yolo-voc-3c.cfg` cuando entrenes.
```
flow --model cfg/tiny-yolo-voc-3c.cfg --load bin/tiny-yolo-voc.weights --train --annotation train/Annotations --dataset train/Images
```
*¿Por qué no debo cambiar el archivo original `tiny-yolo-voc.cfg`?
  
Cuando darkflow ve que estás cargando `tiny-yolo-voc.weights` va a buscar el archivo `tiny-yolo-voc.cfg`en tu folder cfg\ y lo va a comparar con el archivo de configuación nuevo que has puesto con `--model cfg/tiny-yolo-voc-3c.cfg`. En este caso, cada capa tendrá el número mismo número de weights excepto porlos últimos dos, así que cargará los weights en todas las capas hasta las últimas dos porque ahora contienen números diferentes de weights.

# Instalación labelImg

En este caso usaremos LabelImg para crear las cajas delimitadoras. Puedes usar otras herramientas mientras te arrojen el resultado en formato `.xml`.

1. Clona labelImg.
```
git clone sudo https://github.com/tzutalin/labelImg
```

2. Instala pyqt5-dev-tools
```
sudo apt-get install pyqt5-dev-tools
```

3. Instala lxml.
```
sudo apt-get install python3-lxml
```

4. Make qt5py3
```
cd labelImg
make qt5py3
```

5. Compila labelImg.py
```
cd labelImg
python3 labelImg.py
```

El último comando debe abrir el programa.

## Crea tu propio Dataset

Con el botón `open dir` selecciona el directorio donde tienes almacenadas las fotos que usarás para tu dataset. Con el botón `change save dir` selecciona la carpeta donde quieras que se guarden los cambios.

Con cada imagen, pones una caja alrededor del objeto con el botón `create`. Una vez que dibujaste la caja, te pedirá que le asignas una etiqueta, la etiqueta que le pongas debe tener el mismo nombre que asignarás para el archivo `labels.txt`.

Por último, entrena tu modelo con las instrucciones de arriba y ten en cuenta que para entrenar con tu Dataset debes usar alrededor de 2000 steps.
