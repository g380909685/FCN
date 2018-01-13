# FCN for semantic segmentation

This is a simple FCN[1] implementation in Pytorch. There are some differences from the original FCN:

* ResNet features are used
* add dilated convolution
* features of all layers are used
* features of different layers are combined via concatenation instead of summation. 
* VOC training set is argumented by flipping and cropping. 

I haven't test the performance. 

## requirement
pytorch, [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch), tensorboard (for visulization)

If you don't need visulization, then delete the lines about visulization in "main.py".

## Usage
Train:
	
	python main.py

## Reference
[1] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.


