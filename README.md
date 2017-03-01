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

**TODO:** Build the loss function for training.

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


### Evaluation
Follow [this project (TFFRCNN)](https://github.com/CharlesShang/TFFRCNN)
to download and prepare the training, validation, test data. 

Since the program loading the data in `yolo2-pytorch/data` by default,
you can set the data path as following.
```bash
cd yolo2-pytorch
mkdir data
cd data
ln -s $VOCdevkit VOCdevkit2007
```

Set the path of the `trained_model` in `yolo2-pytorch/cfgs/config.py`.
```bash
cd faster_rcnn_pytorch
mkdir output
python test.py
```

###Discuss
I am confused about the difference between YOLO and RPN 
(region proposal network). 

YOLO divides the image into a grid and predicts 
bounding boxes and probabilities for 
each cell in the grid. I think this is what RPN does, 
especially for YOLOv2 which uses a set of anchors for 
each cell. 

One of the main difference between YOLO and RPN 
is the loss functions. YOLO limits predicted location 
coordinates in one of the cells during training and 
testing while RPN associates predicted ROIs with 
ground-truth boxes without any limitation. **Is is enough 
to make a region proposal method become a detector? 
Or I have missed something important.
Welcome to discuss with me.**