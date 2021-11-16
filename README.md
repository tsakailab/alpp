# 知的画像処理＠2021年度先端技術応用講座


## 第1回：深層学習による画像処理の基礎

### 準備
- Google Colaboratory
  - 動作確認
  - Jupyter Notebookについて
- 知的画像処理？
  - 物体検出・認識（detection and classification）：（[参考](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)）
  - カラー化（colorization）：[DeOldify](https://github.com/jantic/DeOldify/blob/master/ImageColorizerColab.ipynb)
  - 画像の変換：[pix2pix](https://phillipi.github.io/pix2pix/)（[参考](https://affinelayer.com/pixsrv/)）<!--（[pix2pix](https://githubtocolab.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)）-->
  - 画風の変換（style transfer）

### 基礎的な画像処理
- 局所特徴の抽出（[参考1](https://setosa.io/ev/image-kernels/)，[参考2]()）
  - 畳み込み演算
  - エッジの検出・強調
  - 色の検出
  - 平均化
- 画像の変換・拡張
  - 幾何学変換（拡大・縮小，回転など）
  - 画素値（色）の変換

---

## 第2回：様々なCNNモデル

### 畳み込みニューラルネットワーク（CNN）の構成要素
- 全結合
- 畳み込み
- バッチノルム
- 活性化
- プーリング

### CNNの構成例
- 画像認識：画像→ラベル（AlexNet，VGG，ResNet）
- 画像処理：画像→画像（エンコーダ・デコーダモデル）
- 学習済みモデルの入手と観察
  - モデルのsummary
  - フィルタの可視化
  - 特徴マップ

### 画像認識の実装に挑戦
- CNNモデルの定義
- 損失関数
- 最適化

---

## 第3回：転移学習

### モデルの学習
- 学習曲線
- ハイパーパラメタの調整

### 学習済みモデルの流用
- モデルの設計
  - 特徴抽出器（backbone）の選択
  - 特徴マップから出力までの設計
- ファインチューニング
- 性能の比較
