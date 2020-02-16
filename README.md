# UCSD CSE253 Progamming assignment 3

Semantic segmentation

We implemented the following network architecture

* FCN
* UNet
* ResNet+ASPP
* UNet (Skip connection) + ASPP
* ResNet+decoder

We utilized yaml file to store the hyperparameter settings for each network. To run the corresponding network, please pass the correct yaml file to `load_config` function in `train.py/test.py`.  

## Configuration file

We use yaml file to store the parameters. An example of ASPP module is shown as below:

```yaml
model: "Deeplab" # specify the model name
loss_method: "cross-entropy"
opt_method: "Adam"
batch_size: 16
img_shape: [512,512]
epochs: 100
num_classes: 19
lr: 0.01 
GPU: True # True if use GPU
backbone: "resnet101" # specify the backbone name to use
save_best: True # Use early stop to store the best model
retrain: False # True if retrain the whole model
retrain_backbone: True # True if fintune the torch backbone like resnet50
use_torch_model: True
CUDA_DIX: [0, 1] # specify the GPU to use
model_save_path: "" # if specified use the this path to save model 
                    # otherwise use the default path to save model
visualize: False # If true generate figures in the test.py script
```



## Train the model

To train the model use the following command. 

```bash
python train.py
```

## Test the model

To test the model use the following command

```bash
python test.py
```

## Models

* In `model/models.py`, we implemented:
  * FCN
  * UNet without batchnormalization
  * UNet with batchnormalization
  * FCN+backbone

* In` model/basic_fcn.py`, we implemented 
  * FCN

* In `model/ASPP.py`, we implemented the Astrous Spatial Pyramid Pooling (AKA ASPP) with 2 versions. 
  * The first version is to use resnet as encoder and ASPP plus a upsampling as classifier. To use this version set `use_torch_model: True` in `config/aspp.yaml`. 

  * The second version is modefied based on our UNet architecture. We use skip connection and encoder-decoder architecture while utilizing ASPP in the decoder part. To use this version set `use_torch_model: False` in `config/aspp.yaml`.

    To train/test this network, please pass `aspp.yaml` to `load_config` in `train.py/test.py`. 

    To just test the shape test for this module, run:

    ```bash
    python model/ASPP.py
    ```

* In `/model/Loss.py`, we implemented the loss function including:
  * cross entropy
  * dice loss
  * WCEloss
  * OhemCELoss
  * SoftmaxFocalLoss

## Utils file

In utils folder, there is some utils functions and dataloader. To test dataloader, run:

```bash
python utils/dataloader.py
```








