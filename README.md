# 知的画像処理＠2021年度先端技術応用講座

この講座の教材は，解説に都合の良いWeb上の記事やデモプログラム等の外部資料，およびサンプルコードで構成しています．
太字の[**colab**](https://github.com/tsakailab/iip/tree/main/colab)は，このGitHubのプロジェクト "iip" で講師が提供するサンプルコードをGoogle Colaboratoryで開くリンクです．

## 第1回：深層学習による画像処理の基礎

### 準備
- [Google Colaboratory](https://colab.research.google.com/)（[参考](https://blog.kikagaku.co.jp/google-colab-howto)）
  - 動作確認
  - Jupyter Notebookについて
- 手軽に動かせる知的画像処理(?)の例
  - [Torch Hub](https://pytorch.org/hub/research-models)
  - 画像認識（image recognition）：[ResNet50 (colab)](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/nvidia_deeplearningexamples_resnet50.ipynb)
  - 物体検出・識別（object detection and classification）：[SSD](https://arxiv.org/pdf/1512.02325.pdf)（[colab](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/nvidia_deeplearningexamples_ssd.ipynb)，[参考1](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf)，[参考2](https://jonathan-hui.medium.com/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06)），[参考3](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)
  - カラー化（colorization）：[DeOldify (colab)](https://github.com/jantic/DeOldify/blob/master/ImageColorizerColab.ipynb)
  - 画像の変換：[pix2pix](https://phillipi.github.io/pix2pix/)（[参考](https://affinelayer.com/pixsrv/)）<!--（[pix2pix](https://githubtocolab.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)）-->
  - 画風の変換：[Style Transfer (colab)](https://githubtocolab.com/tsakailab/iip/blob/main/sandbox/NeuralStyleTransfer.ipynb)，[MSG-Net (colab)](https://colab.research.google.com/github/zhanghang1989/PyTorch-Multi-Style-Transfer/blob/master/msgnet.ipynb)
  - その他：[colabで動く最新技術のリンク集](https://github.com/amrzv/awesome-colab-notebooks)

### 基礎的な画像処理
- 大域的な変換 [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_global_operations.ipynb)（[参考](https://pytorch.org/vision/stable/transforms.html)）
  - 幾何学変換（拡大・縮小，回転など）
  - 画素値（色）の変換
- 畳み込みによる局所演算 [**colab**](https://githubtocolab.com/tsakailab/iip/blob/main/colab/iip_local_operations.ipynb)
（[参考1](https://setosa.io/ev/image-kernels/)，[参考2](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)）
  - エッジの検出・強調
  - 色の検出
  - 平均化

---

## 第2回：様々なCNNモデル

### 畳み込みニューラルネットワーク（CNN）の構成要素
- 全結合
- 畳み込み
- [バッチノルム](https://arxiv.org/abs/1502.03167)（[参考1](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)，[参考2](https://theaisummer.com/normalization/)）
- 活性化
- プーリング

### CNNの構成例
- 画像認識：画像→ラベル（[AlexNet，VGG，ResNetなど](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-image-classification-5fdbca4a05e2)）
- 画像処理：画像→画像（エンコーダ・デコーダモデル）
- 学習済みモデルの入手と観察
  - モデルのsummary
  - フィルタの可視化（[参考](https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa843e74e)）
  - 特徴マップ

### CNNの実装に挑戦<!-- CNNで小さなAEを作ってフィルタカーネルを観察する -->
- CNNモデルの定義
- 損失関数
- 最適化

---

## 第3回：転移学習

### モデルの学習
- 学習曲線
- ハイパーパラメタの調整

### 学習済みモデルの流用
- 事前学習済みモデル<!-- https://note.nkmk.me/python-pytorch-hub-torchvision-models/ -->
  - [PyTorch Hub](https://pytorch.org/hub/)
  - [torchvision.models](https://pytorch.org/vision/stable/models.html)
- モデルの設計
  - 特徴抽出器（backbone）の選択
  - 特徴マップから出力までの設計
- ファインチューニング
- 性能の比較
