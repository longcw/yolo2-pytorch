# YOLOv2 in PyTorch
This is a [PyTorch](https://github.com/pytorch/pytorch)
implementation of YOLOv2.
This project is mainly based on [darkflow](https://github.com/thtrieu/darkflow)
and [darknet](https://github.com/pjreddie/darknet).

For details about YOLO and YOLOv2 please refer to their [project page](https://pjreddie.com/darknet/yolo/) 
and the [paper](https://arxiv.org/abs/1612.08242):
YOLO9000: Better, Faster, Stronger by Joseph Redmon and Ali Farhadi.

I used a Cython extension for postprocessing and 
`multiprocessing.Pool` for image preprocessing.
Testing an image in VOC2007 costs about 13~20ms. 

### Installation and demo
1. Clone this repository
    ```bash
    git clone git@github.com:longcw/yolo2-pytorch.git
    ```

2. Build the reorg layer ([`tf.extract_image_patches`](https://www.tensorflow.org/api_docs/python/tf/extract_image_patches))
    ```bash
    cd yolo2-pytorch
    ./make.sh
    ```
3. Download the trained model [yolo-voc.weights.h5](https://drive.google.com/open?id=0B4pXCfnYmG1WUUdtRHNnLWdaMEU) 
and set the model path in `demo.py`
4. Run demo `python demo.py`. 

### Training YOLOv2
You can train YOLO2 on any dataset. Here we train it on VOC2007/2012.

1. Download the training, validation, test data and VOCdevkit

    ```bash
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    ```

2. Extract all of these tars into one directory named `VOCdevkit`

    ```bash
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_08-Jun-2007.tar
    ```

3. It should have this basic structure

    ```bash
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
    # ... and several other directories ...
    ```
    
4. Since the program loading the data in `yolo2-pytorch/data` by default,
you can set the data path as following.
    ```bash
    cd yolo2-pytorch
    mkdir data
    cd data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    
5. Download the [pretrained darknet19 model](https://drive.google.com/file/d/0B4pXCfnYmG1WRG52enNpcV80aDg/view?usp=sharing)
and set the path in `yolo2-pytorch/cfgs/exps/darknet19_exp1.py`.

7. (optional) Training with TensorBoard.

    To use the TensorBoard, install Crayon (https://github.com/torrvision/crayon)
and set `use_tensorboard = True` in `yolo2-pytorch/cfgs/config.py`.


6. Run the training program: `python train.py`.



### Evaluation

Set the path of the `trained_model` in `yolo2-pytorch/cfgs/config.py`.
```bash
cd faster_rcnn_pytorch
mkdir output
python test.py
```
