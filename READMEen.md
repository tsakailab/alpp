# Introduction to Deep Learning for Image Processing
A part of Course Materials for Advanced Lectures on Pattern Processing, Graduate School of Engineering, Nagasaki University, 2022 ([Japanese](README.md))

<br>
The course materials listed here consist of articles on the Web, external materials such as demo programs, and sample codes that are handy for explaining deep learning. 
The [**colab**](https://github.com/tsakailab/alpp/tree/main/colab) in bold is a link to open in Google Colaboratory a sample code at this GitHub repository "alpp" provided by the instructor.


---

## Part 1: Fundamentals of Image Processing by Deep Learning

### Preliminaries
- [Google Colaboratory](https://colab.research.google.com/)（[cf.](https://blog.kikagaku.co.jp/google-colab-howto)）
  - About Jupyter Notebook
  - Check the operating environment [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/display_colab_spec.ipynb)
- Colab-ready sample codes of image processing
  - [Torch Hub](https://pytorch.org/hub/research-models)
  - Image recognition: [ResNet50 (colab)](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/nvidia_deeplearningexamples_resnet50.ipynb)
  - Object detection and classification: [SSD](https://arxiv.org/pdf/1512.02325.pdf) ([colab](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/nvidia_deeplearningexamples_ssd.ipynb), [cf. 1](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf), [cf. 2](https://jonathan-hui.medium.com/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06), [cf. 3](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852))
  - Colorization: [DeOldify (colab)](https://github.com/jantic/DeOldify/blob/master/ImageColorizerColab.ipynb)
  - [pix2pix](https://phillipi.github.io/pix2pix/) ([cf.](https://affinelayer.com/pixsrv/))<!-- ([pix2pix](https://githubtocolab.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb))-->
  - [Style Transfer (**colab**)](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/NeuralStyleTransfer.ipynb), [MSG-Net (colab)](https://colab.research.google.com/github/zhanghang1989/PyTorch-Multi-Style-Transfer/blob/master/msgnet.ipynb)
  - Misc.: [Awesome colab notebooks collection for ML experiments](https://github.com/amrzv/awesome-colab-notebooks). [the-incredible-pytorch](https://www.ritchieng.com/the-incredible-pytorch/)

### Basic Image Processing
- Global transformations [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/alpp_global_operations.ipynb) ([cf.](https://pytorch.org/vision/stable/transforms.html)）
  - Geometric transformations (scaling, rotation, etc.)
  - Color deformation (pixel value transformation)
- Local operations by convolution [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/alpp_local_operations.ipynb)
 ([cf. 1](https://setosa.io/ev/image-kernels/), [cf. 2](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1))
  - Edge detection and enhancement
  - Color detection
  - Averaging and blurring

---

## Part 2: Various CNN models

### Architectures of Convolutional Neural Networks (CNN)
- Image recognition: image to label ([AlexNet, VGG, ResNet, etc.](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-image-classification-5fdbca4a05e2))
- Image processing: image to image (encoder-decoder model: [cf. 1](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html), [cf. 2](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#types-of-generative-models), hourglass networks: [cf. 1](https://en.wikipedia.org/wiki/U-Net), [cf. 2](https://medium.com/@sunnerli/simple-introduction-about-hourglass-like-model-11ee7c30138), [cf. 3](http://ais.informatik.uni-freiburg.de/teaching/ss19/deep_learning_lab/presentation_lectureCV.pdf))
- Obtaining and observing pre-trained models
  - Model summary [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/alpp_model_summary.ipynb)
  - Filter visualization [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/alpp_model_visualize_conv_kernels.ipynb) ([cf. 1](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf#page=7), [cf. 2](https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e))
  - Feature maps [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/alpp_model_visualize_featuremaps.ipynb) ([cf.](https://github.com/utkuozbulak/pytorch-cnn-visualizations))
- Class activation mapping (CAM)
  - Partial occlusion ([cf.](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf#page=10)) / self-made in forward prop. ([CAM via GAP](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)) / self-made by backprop. [Grad-CAM](https://arxiv.org/pdf/1610.02391.pdf) / linear approximation [LIME](https://arxiv.org/pdf/1602.04938.pdf)

### Components of CNN
- [cf. 1](https://en.wikipedia.org/wiki/Convolutional_neural_network), [cf. 2](https://www.electricalelibrary.com/en/2018/11/20/what-are-convolutional-neural-networks/), [cf. 3](https://www.researchgate.net/figure/Overview-and-details-of-a-convolutional-neural-network-CNN-architecture-for-image_fig2_341576780)
- [Convolutional layer](https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer)
- [Batch normalization layer (bn)](https://arxiv.org/abs/1502.03167)：[cf. 1](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html), [cf. 2](https://theaisummer.com/normalization/)
- [Activation layer](https://en.wikipedia.org/wiki/Activation_function)
- [Pooling layer](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layers): [cf. 1](https://pytorch.org/docs/stable/nn.html#pooling-layers), [cf. 2](https://arxiv.org/ftp/arxiv/papers/2009/2009.07485.pdf)
- [Fully connected layer (fc/linear)](https://en.wikipedia.org/wiki/Convolutional_neural_network#Fully_connected_layers): [cf.](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)


---

## Part 3: Implementation and Training of CNN Models

### Implementation [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/alpp_cnn_practice.ipynb)
- Model definition
- Loss functions
- Optimization

### Training [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/alpp_cnn_practice.ipynb)
- Hyperparameter tuning
- Learning curves
  - Overfitting and early stopping
- Observation of CAM [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/alpp_model_cam.ipynb)
  - [CAM via GAP](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)


---

## Part 4: Deep Learning with Small Dataset

### Image Recognition by Transfer Learning [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/alpp_cnn_practice_transfer_learning.ipynb)
- Pre-trained models<!-- https://note.nkmk.me/python-pytorch-hub-torchvision-models/ -->
  - [PyTorch Hub](https://pytorch.org/hub/)
  - [torchvision.models](https://pytorch.org/vision/stable/models.html)
- Model design
  - Feature extractor (backbone)
  - Classifier (from feature maps to output)
- Fine tuning
- Observation of CAM [**colab**](https://githubtocolab.com/tsakailab/alpp/blob/main/colab/alpp_model_cam.ipynb)

---

## Miscellaneous
### Computer Vision Resources
- [Kaggle](https://www.kaggle.com/)
  - [Courses](https://www.kaggle.com/learn) , e.g., [Computer Vision](https://www.kaggle.com/learn/computer-vision)
  - [Datasets](https://www.kaggle.com/datasets), e.g., [Computer Vision](https://www.kaggle.com/datasets?tags=13207-Computer+Vision)
- [CVF Open Access](https://openaccess.thecvf.com/menu)
  - [CVPR](https://en.wikipedia.org/wiki/Conference_on_Computer_Vision_and_Pattern_Recognition)
  - [cvpaper.challenge](http://xpaperchallenge.org/cv/)
- Research groups
  - [VGG](https://www.robots.ox.ac.uk/~vgg/)
### Easy-to-use datasets
- [TensorFlowDatasets](https://github.com/tensorflow/datasets)
