
# Are Your Images Used for Training? Auditing Data Provenance in Real-world Text-to-image Diffusion Models


## Quick Start

### Requirements
* Python 3.8.16
* pytorch 1.12.1+cu113
* numpy 1.20.3
* torchvision 0.13.1+cu113
* scikit-learn 1.2.2
* diffusers 0.17.1
* clip
* datetime
* pathlib
* PIL

### Quick Start

#### Download [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and [LAION-mi dataset](https://drive.google.com/drive/folders/17lRvzW4uXDoCf1v_sIiaMnKGIARVunNU)


#### Extract text and image features

```
sh sh_extract.sh
```

#### Train and evaluate FSCA auditing model 

```
sh fsca_command/sh_train_0.sh
```



