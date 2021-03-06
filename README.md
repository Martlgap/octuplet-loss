# Octuplet Loss - Make Face Recognition Robust Against Image Resolution
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://img.shields.io/badge/license-MIT-blue)
[![Last Commit](https://img.shields.io/github/last-commit/martlgap/octuplet-loss)](https://img.shields.io/github/last-commit/martlgap/octuplet-loss)


Here, we release our code utilized in the following paper:
- [Octuplet Loss - Make Face Recognition Robust Against Image Resolution
](TBD)

![Loss Visualization](https://github.com/martlgap/octuplet-loss/blob/main/loss_vis.jpg?raw=true)

## 🏆 Performance (Accuracy [%])
| Model | [LFW](http://vis-www.cs.umass.edu/lfw/) | [XQLFW](https://martlgap.github.io/xqlfw/) |
|---|---|---|
| [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf) | 99.55 | 93.27 |
| [MagFace](https://openaccess.thecvf.com/content/CVPR2021/papers/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.pdf) | 99.63 | 92.92 |
| [FaceTransformer](https://arxiv.org/abs/2103.14803) | 99.73 | 95.12 |

## 💻 Code
We provide the code of our proposed octuplet loss for [tf_octuplet_loss.py (Tensorflow 2)](https://github.com/martlgap/octuplet-loss/blob/main/tf_octuplet_loss.py) 
and [pt_octuplet_loss.py (PyTorch)](https://github.com/martlgap/octuplet-loss/blob/main/pt_octuplet_loss.py). 


## 🥣 Requirements
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)](https://img.shields.io/badge/Python-3.8-blue)


## 📖 Cite
If you use our code please consider citing:
~~~tex
@misc{https://doi.org/10.48550/arxiv.2207.06726,
  doi = {10.48550/ARXIV.2207.06726},
  url = {https://arxiv.org/abs/2207.06726},
  author = {Knoche, Martin and Elkadeem, Mohamed and Hörmann, Stefan and Rigoll, Gerhard},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Octuplet Loss: Make Face Recognition Robust to Image Resolution},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
~~~


## ✉️ Contact
For any inquiries, please open an [issue](https://github.com/Martlgap/octuplet-loss/issues) on GitHub or send an E-Mail to: [Martin.Knoche@tum.de](mailto:Martin.Knoche@tum.de)
