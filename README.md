# AttCNN-with-Focal-Loss
This repository includes ATTCnn code implementation of this paper: https://www.cs.umd.edu/~emhand/Papers/AAAI2018_SelectiveLearning.pdf

## Improvements

Instead of using selective learning, here contains an implementation with focal loss which does the same task and weighs down the more easily classified examples and weighs more the hard to classify classes. This way Deep Neural Net will focus more on the hard to classify classes which i propose will be the ones with less images in dataset. Paper mentioned proved that batch balancing led to better accuracy. Hence, it is fair to assume that with classes with less images were the reason behind poor accuracy. Focal loss will focus on hard to classify classes which are the ones with less images and should produce better accuracy.
