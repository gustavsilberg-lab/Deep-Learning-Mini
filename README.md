---
language:
- en
size_categories:
- 10K<n<100K
task_categories:
- image-classification
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': red and white circle 20 kph speed limit
          '1': red and white circle 30 kph speed limit
          '2': red and white circle 50 kph speed limit
          '3': red and white circle 60 kph speed limit
          '4': red and white circle 70 kph speed limit
          '5': red and white circle 80 kph speed limit
          '6': end / de-restriction of 80 kph speed limit
          '7': red and white circle 100 kph speed limit
          '8': red and white circle 120 kph speed limit
          '9': red and white circle red car and black car no passing
          '10': red and white circle red truck and black car no passing
          '11': red and white triangle road intersection warning
          '12': white and yellow diamond priority road
          '13': red and white upside down triangle yield right-of-way
          '14': stop
          '15': empty red and white circle
          '16': red and white circle no truck entry
          '17': red circle with white horizonal stripe no entry
          '18': red and white triangle with exclamation mark warning
          '19': red and white triangle with black left curve approaching warning
          '20': red and white triangle with black right curve approaching warning
          '21': red and white triangle with black double curve approaching warning
          '22': red and white triangle rough / bumpy road warning
          '23': red and white triangle car skidding / slipping warning
          '24': red and white triangle with merging / narrow lanes warning
          '25': red and white triangle with person digging / construction / road work
            warning
          '26': red and white triangle with traffic light approaching warning
          '27': red and white triangle with person walking warning
          '28': red and white triangle with child and person walking warning
          '29': red and white triangle with bicyle warning
          '30': red and white triangle with snowflake / ice warning
          '31': red and white triangle with deer warning
          '32': white circle with gray strike bar no speed limit
          '33': blue circle with white right turn arrow mandatory
          '34': blue circle with white left turn arrow mandatory
          '35': blue circle with white forward arrow mandatory
          '36': blue circle with white forward or right turn arrow mandatory
          '37': blue circle with white forward or left turn arrow mandatory
          '38': blue circle with white keep right arrow mandatory
          '39': blue circle with white keep left arrow mandatory
          '40': blue circle with white arrows indicating a traffic circle
          '41': white circle with gray strike bar indicating no passing for cars has
            ended
          '42': white circle with gray strike bar indicating no passing for trucks
            has ended
  splits:
  - name: train
    num_bytes: 252930879.36
    num_examples: 26640
  - name: test
    num_bytes: 104816357.02
    num_examples: 12630
  - name: contrast
    num_bytes: 104816357.02
    num_examples: 12630
  - name: gaussian_noise
    num_bytes: 104816357.02
    num_examples: 12630
  - name: impulse_noise
    num_bytes: 104816357.02
    num_examples: 12630
  - name: jpeg_compression
    num_bytes: 104816357.02
    num_examples: 12630
  - name: motion_blur
    num_bytes: 104816357.02
    num_examples: 12630
  - name: pixelate
    num_bytes: 39121740.4
    num_examples: 12630
  - name: spatter
    num_bytes: 104816357.02
    num_examples: 12630
  download_size: 1027074522
  dataset_size: 1025767118.8999999
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: contrast
    path: data/contrast-*
  - split: gaussian_noise
    path: data/gaussian_noise-*
  - split: impulse_noise
    path: data/impulse_noise-*
  - split: jpeg_compression
    path: data/jpeg_compression-*
  - split: motion_blur
    path: data/motion_blur-*
  - split: pixelate
    path: data/pixelate-*
  - split: spatter
    path: data/spatter-*
---



# Dataset Card for German Traffic Sign Recognition Benchmark

This dataset contains images of 43 classes of traffic signs. It is intended for developing and benchmarking traffic sign recognition systems.

## Dataset Details

### Dataset Description

The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class classification dataset featuring 43 classes of traffic signs. 
The images were cropped from a larger set of images to focus on the traffic sign and eliminate background. 
Multiple data augmentations such as Gaussian noise, motion blur, contrast changes, etc. are provided as additional test sets to benchmark model robustness.

### Dataset Sources

- [Paper with code](https://paperswithcode.com/dataset/gtsrb)

## Uses

### Direct Use

```python
from datasets import load_dataset

dataset = load_dataset('tanganke/gtsrb')
```

## Dataset Structure

The dataset is provided in 9 splits, including training data and clean test data:

- train: 26,640 images 
- test: 12,630 images

and 7 kinds of corrupted test datasets to evaluate the robustness:

- contrast: 12,630 contrast-adjusted test images  
- gaussian_noise: 12,630 Gaussian noise augmented test images
- impulse_noise: 12,630 impulse noise augmented test images 
- jpeg_compression: 12,630 JPEG-compressed test images
- motion_blur: 12,630 motion-blurred test images
- pixelate: 12,630 pixelated test images
- spatter: 12,630 spatter augmented test images

Each split contains 43 classes of traffic signs, with the class labels and names specified in the dataset metadata.

## Citation [optional]

You can use any of the provided BibTeX entries for your reference list:

```bibtex
@article{stallkampManVsComputer2012,
  title = {Man vs. Computer: {{Benchmarking}} Machine Learning Algorithms for Traffic Sign Recognition},
  shorttitle = {Man vs. Computer},
  author = {Stallkamp, J. and Schlipsing, M. and Salmen, J. and Igel, C.},
  year = {2012},
  month = aug,
  journal = {Neural Networks},
  series = {Selected {{Papers}} from {{IJCNN}} 2011},
  volume = {32},
  pages = {323--332},
  issn = {0893-6080},
  doi = {10.1016/j.neunet.2012.02.016},
  url = {https://www.sciencedirect.com/science/article/pii/S0893608012000457},
  keywords = {Benchmarking,Convolutional neural networks,Machine learning,Traffic sign recognition}
}

@misc{yangAdaMergingAdaptiveModel2023,
  title = {{{AdaMerging}}: {{Adaptive Model Merging}} for {{Multi-Task Learning}}},
  shorttitle = {{{AdaMerging}}},
  author = {Yang, Enneng and Wang, Zhenyi and Shen, Li and Liu, Shiwei and Guo, Guibing and Wang, Xingwei and Tao, Dacheng},
  year = {2023},
  month = oct,
  number = {arXiv:2310.02575},
  eprint = {2310.02575},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2310.02575},
  url = {http://arxiv.org/abs/2310.02575},
  archiveprefix = {arxiv},
  keywords = {Computer Science - Computer Vision and Pattern Recognition,Computer Science - Machine Learning}
}

@misc{tangConcreteSubspaceLearning2023,
  title = {Concrete {{Subspace Learning}} Based {{Interference Elimination}} for {{Multi-task Model Fusion}}},
  author = {Tang, Anke and Shen, Li and Luo, Yong and Ding, Liang and Hu, Han and Du, Bo and Tao, Dacheng},
  year = {2023},
  month = dec,
  number = {arXiv:2312.06173},
  eprint = {2312.06173},
  publisher = {arXiv},
  url = {http://arxiv.org/abs/2312.06173},
  archiveprefix = {arxiv},
  copyright = {All rights reserved},
  keywords = {Computer Science - Machine Learning}
}


@misc{tangMergingMultiTaskModels2024,
  title = {Merging {{Multi-Task Models}} via {{Weight-Ensembling Mixture}} of {{Experts}}},
  author = {Tang, Anke and Shen, Li and Luo, Yong and Yin, Nan and Zhang, Lefei and Tao, Dacheng},
  year = {2024},
  month = feb,
  number = {arXiv:2402.00433},
  eprint = {2402.00433},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2402.00433},
  url = {http://arxiv.org/abs/2402.00433},
  archiveprefix = {arxiv},
  copyright = {All rights reserved},
  keywords = {Computer Science - Computer Vision and Pattern Recognition,Computer Science - Machine Learning}
}
```


## Dataset Card Authors

Anke Tang

## Dataset Card Contact

[tang.anke@foxmail.com](mailto:tang.anke@foxmail.com)