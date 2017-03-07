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

### Training YOLOv2
For training yolo2, you can get darknet19 model pretrained on ImageNet 
by the author in [darknet19.weights.npz](https://drive.google.com/file/d/0B4pXCfnYmG1WRG52enNpcV80aDg/view?usp=sharing)
and load it by `net.load_from_npz(cfg.pretrained_model, num_conv=18)`,
and then prepare the dataset and run `python train.py`

I implemented the loss function as same as [darknet](https://github.com/pjreddie/darknet)
and trained it on VOC2007. The mAP on VOC2007 test set is 0.7186 after 158 epoch.