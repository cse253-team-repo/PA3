# UCSD CSE253 Progamming assignment 3

Semantic segmentation

We implemented the following network architecture

* FCN
* UNet
* ResNet+ASPP
* UNet+ASPP
* ResNet+decoder

We utilized yaml file to store the highper parameter settings for each network. To run the corresponding network, please pass the correct yaml file to `load_config` function in `train.py/test.py`.  


## ASPP module
In model/ASPP.py, we implemented the Astrous Spatial Pyramid Pooling (AKA ASPP) with 2 versions. 

* The first version is to use resnet as encoder and ASPP plus a upsampling as classifier. To use this version set `use_torch_model: True` in `config/aspp.yaml`. 
* The second version is modefied based on our UNet architecture. We use skip connection and encoder-decoder architecture while utilizing ASPP in the decoder part. To use this version set `use_torch_model: False` in `config/aspp.yaml`.

To train/test this network, please pass `aspp.yaml` to `load_config` in `train.py/test.py`. 

To just test the shape test for this module, run:

```bash
python model/ASPP.py
```

















Unet bn
[0.76558155 0.57691383 0.77113068 0.12833847 0.22157449 0.45102286
 0.36401013 0.55917227 0.87275022 0.43038559 0.81529766 0.46183452
 0.15253368 0.77989101 0.04093395 0.11353317 0.01616228 0.09720594
 0.48719034]
valid accuracy: 0.9106153805324106 	 valid ious 0.4266032968696795
