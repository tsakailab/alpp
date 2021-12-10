# 知的画像処理＠2021年度先端技術応用講座

この講座の教材は，解説に都合の良いWeb上の記事やデモプログラム等の外部資料，およびサンプルコードで構成しています．
太字の[**colab**](https://github.com/tsakailab/iip/tree/main/colab)は，このGitHubのプロジェクト "iip" で講師が提供するサンプルコードをGoogle Colaboratoryで開くリンクです．

## 第1回：深層学習による画像処理の基礎

### 準備
- [Google Colaboratory](https://colab.research.google.com/)（[参考](https://blog.kikagaku.co.jp/google-colab-howto)）
  - Jupyter Notebookについて
  - 動作環境の確認 [**colab**](https://github.com/tsakailab/iip/blob/main/colab/display_colab_spec.ipynb)
- 手軽に動かせる知的画像処理(?)の例
  - [Torch Hub](https://pytorch.org/hub/research-models)
  - 画像認識（image recognition）：[ResNet50 (colab)](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/nvidia_deeplearningexamples_resnet50.ipynb)
  - 物体検出・識別（object detection and classification）：[SSD](https://arxiv.org/pdf/1512.02325.pdf)（[colab](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/nvidia_deeplearningexamples_ssd.ipynb)，[参考1](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf)，[参考2](https://jonathan-hui.medium.com/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06)），[参考3](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)
  - カラー化（colorization）：[DeOldify (colab)](https://github.com/jantic/DeOldify/blob/master/ImageColorizerColab.ipynb)
  - 画像の変換：[pix2pix](https://phillipi.github.io/pix2pix/)（[参考](https://affinelayer.com/pixsrv/)）<!--（[pix2pix](https://githubtocolab.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)）-->
  - 画風の変換：[Style Transfer (**colab**)](https://githubtocolab.com/tsakailab/iip/blob/main/colab/NeuralStyleTransfer.ipynb)，[MSG-Net (colab)](https://colab.research.google.com/github/zhanghang1989/PyTorch-Multi-Style-Transfer/blob/master/msgnet.ipynb)
  - その他：[colabで動く最新技術のリンク集](https://github.com/amrzv/awesome-colab-notebooks)，[the-incredible-pytorch](https://www.ritchieng.com/the-incredible-pytorch/)

### 基礎的な画像処理
- 大域的な変換 [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_global_operations.ipynb)（[参考](https://pytorch.org/vision/stable/transforms.html)）
  - 幾何学変換（拡大・縮小，回転など）
  - 画素値（色）の変換
- 畳み込みによる局所演算 [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_local_operations.ipynb)
（[参考1](https://setosa.io/ev/image-kernels/)，[参考2](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)）
  - エッジの検出・強調
  - 色の検出
  - 平均化・ぼかし

---

## 第2回：様々なCNNモデル

### 畳み込みニューラルネットワーク（CNN）の構成例
- 画像認識：画像→ラベル（[AlexNet，VGG，ResNetなど](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-image-classification-5fdbca4a05e2)）
- 画像処理：画像→画像（エンコーダ・デコーダモデル）[参考1](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html)，[参考2](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#types-of-generative-models)
- 学習済みモデルの入手と観察
  - モデルのsummary [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_model_summary.ipynb)
  - フィルタの可視化 [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_model_visualize_conv_kernels.ipynb)（[参考1](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf#page=7)，[参考2](https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e)）
  - 特徴マップ [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_model_visualize_featuremaps.ipynb)（[参考](https://github.com/utkuozbulak/pytorch-cnn-visualizations)）
- クラス活性化マッピング
  - 部分的に隠す（[参考](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf#page=10)）
  - モデルに作らせる（[GAPからCAMを作る](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)）
  - 逆伝播で算出する [Grad-CAM](https://arxiv.org/pdf/1610.02391.pdf)
  - 線形モデルで近似する [LIME](https://arxiv.org/pdf/1602.04938.pdf)

### CNNの構成要素
- 畳み込み
- [バッチノルム](https://arxiv.org/abs/1502.03167)（[参考1](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)，[参考2](https://theaisummer.com/normalization/)）
- 活性化
- プーリング（[参考](https://arxiv.org/ftp/arxiv/papers/2009/2009.07485.pdf)）
- 全結合

### CNNの実装に挑戦 [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_cnn_practice.ipynb)<!-- CNNで小さなAEを作ってフィルタカーネルを観察する -->
- CNNモデルの定義
- 損失関数
- 最適化

---

## 第3回：転移学習

### モデルの学習 [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_cnn_practice.ipynb)
- ハイパーパラメータの調整
- 学習曲線
  - 過学習（overfitting）と早期打ち切り（early stopping）
- CAMの観察 [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_model_cam.ipynb)
  - [GAPからCAMを作る](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)

### 少数データの画像認識 [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_cnn_practice_transfer_learning.ipynb)
- 流用できる事前学習済みモデル<!-- https://note.nkmk.me/python-pytorch-hub-torchvision-models/ -->
  - [PyTorch Hub](https://pytorch.org/hub/)
  - [torchvision.models](https://pytorch.org/vision/stable/models.html)
- モデルの設計
  - 特徴抽出器（backbone features）の選択
  - 特徴マップから出力までの設計
- ファインチューニング
- CAMの観察 [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_model_cam.ipynb)

## その他
- [Kaggle](https://www.kaggle.com/)
  - [Courses](https://www.kaggle.com/learn) , e.g., [Computer Vision](https://www.kaggle.com/learn/computer-vision)
  - [Datasets](https://www.kaggle.com/datasets), e.g., [Computer Vision](https://www.kaggle.com/datasets?tags=13207-Computer+Vision)
- [CVF Open Access](https://openaccess.thecvf.com/menu)
  - [CVPR](https://en.wikipedia.org/wiki/Conference_on_Computer_Vision_and_Pattern_Recognition)
  - [cvpaper.challenge](http://xpaperchallenge.org/cv/)
