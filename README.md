# Emoji Prediction

Keras implementation of the paper **Which Image Suits best for my Image?**.

**Which Image Suits best for my Image?**\
[Anurag Illendula*](https://github.com/aianurag09), [KV Manohar*](https://kvmanohar22.github.io), Manish Reddy Yedulla\
*Department of Mathematics, IIT Kharagpur and Department of Engineering Sciences, IIT Hyderabad\
At 2018 IEEE/WIC/ACM International Conference on Web Intelligence (WI '18)



## Prerequisites
- Ubuntu 16.04  (tested only on Linux)
- Python 2.7
- Keras
- OpenCV


## Usage

- Generating fasttext model


- Available options
   ```bash
   $ python main.py --help 
   Using TensorFlow backend.
  usage: main.py [-h] [-i IMG_DIR] [-w WEIGHTS_PATH]

  Emoji Prediction

  optional arguments:
    -h, --help            show this help message and exit
    -i IMG_DIR, --img_dir IMG_DIR
                          Directory containing test images
    -w WEIGHTS_PATH, --weights_path WEIGHTS_PATH
                          Path to weights file
   ```

## Results

The following table contains emojis predicted from our model.


| Images                                                               | Text description           | Using sense definition  | Using emoji senses |
| -------------------------------------------------------------------- |:--------------------------:|:-----------------------:|:-------------------:|
| <img src="imgs/COCO_train2014_000000000086.jpg" height="224" width="224"> | A person looksdown at something while sitting on a bike | <img src="emojis/i1/230.png" height="50" width="50"> <img src="emojis/i1/48.png" height="50" width="50"> <img src="emojis/i1/162.png" height="50" width="50"> <img src="emojis/i1/773.png" height="50" width="50"> <img src="emojis/i1/214.png" height="50" width="50">                     | <img src="emojis/j1/821.png" height="50" width="50"> <img src="emojis/j1/214.png" height="50" width="50"> <img src="emojis/j1/1056.png" height="50" width="50"> <img src="emojis/j1/456.png" height="50" width="50"> <img src="emojis/j1/827.png" height="50" width="50">                   |
| <img src="imgs/COCO_train2014_000000000307.jpg" height="224" width="224"> | The dog is playing with histoy in the grass | <img src="emojis/i2/1849.png" height="50" width="50"> <img src="emojis/i2/1829.png" height="50" width="50"> <img src="emojis/i2/1883.png" height="50" width="50"> <img src="emojis/i2/1848.png" height="50" width="50"> <img src="emojis/i2/1924.png" height="50" width="50">                  | <img src="emojis/j2/1829.png" height="50" width="50"> <img src="emojis/j2/1848.png" height="50" width="50"> <img src="emojis/j2/1883.png" height="50" width="50"> <img src="emojis/j2/1922.png" height="50" width="50"> <img src="emojis/j2/1825.png" height="50" width="50">                   |
| <img src="imgs/COCO_train2014_000000000431.jpg" height="224" width="224"> | A tennis player in action on the court | <img src="emojis/i3/2089.png" height="50" width="50"> <img src="emojis/i3/4.png" height="50" width="50"> <img src="emojis/i3/2086.png" height="50" width="50"> <img src="emojis/i3/108.png" height="50" width="50"> <img src="emojis/i3/2057.png" height="50" width="50">                  | <img src="emojis/j3/2089.png" height="50" width="50"> <img src="emojis/j3/2011.png" height="50" width="50"> <img src="emojis/j3/167.png" height="50" width="50"> <img src="emojis/j3/1637.png" height="50" width="50"> <img src="emojis/j3/387.png" height="50" width="50">                   |
| <img src="imgs/COCO_train2014_000000000605.jpg" height="224" width="224"> | Cup of coffee with dessert items on a wooden grained table |  <img src="emojis/i4/2234.png" height="50" width="50"> <img src="emojis/i4/2274.png" height="50" width="50"> <img src="emojis/i4/2247.png" height="50" width="50"> <img src="emojis/i4/52.png" height="50" width="50"> <img src="emojis/i4/2259.png" height="50" width="50">                  | <img src="emojis/j4/2207.png" height="50" width="50"> <img src="emojis/j4/2235.png" height="50" width="50"> <img src="emojis/j4/2236.png" height="50" width="50"> <img src="emojis/j4/2221.png" height="50" width="50"> <img src="emojis/j4/1953.png" height="50" width="50">                   |
| <img src="imgs/COCO_train2014_000000000643.jpg" height="224" width="224"> | A desktop computer and monitor on a desk |  <img src="emojis/i5/350.png" height="50" width="50"> <img src="emojis/i5/1079.png" height="50" width="50"> <img src="emojis/i5/1077.png" height="50" width="50"> <img src="emojis/i5/1548.png" height="50" width="50"> <img src="emojis/i5/1525.png" height="50" width="50">                  |  <img src="emojis/j5/1079.png" height="50" width="50"> <img src="emojis/j5/1077.png" height="50" width="50"> <img src="emojis/j5/538.png" height="50" width="50"> <img src="emojis/j5/1548.png" height="50" width="50"> <img src="emojis/j5/2094.png" height="50" width="50">                  |
| <img src="imgs/COCO_train2014_000000000828.jpg" height="224" width="224"> | A man with a paddle riding a wave on a surf board |  <img src="emojis/i6/133.png" height="50" width="50"> <img src="emojis/i6/581.png" height="50" width="50"> <img src="emojis/i6/181.png" height="50" width="50"> <img src="emojis/i6/1871.png" height="50" width="50"> <img src="emojis/i6/707.png" height="50" width="50">                  |  <img src="emojis/j6/1094.png" height="50" width="50"> <img src="emojis/j6/1650.png" height="50" width="50"> <img src="emojis/j6/581.png" height="50" width="50"> <img src="emojis/j6/1247.png" height="50" width="50"> <img src="emojis/j6/820.png" height="50" width="50">                  |
| <img src="imgs/COCO_train2014_000000000853.jpg" height="224" width="224"> | A sandwich covered in red sac use and chicken salad |  <img src="emojis/i7/50.png" height="50" width="50"> <img src="emojis/i7/2274.png" height="50" width="50"> <img src="emojis/i7/2247.png" height="50" width="50"> <img src="emojis/i7/2259.png" height="50" width="50"> <img src="emojis/i7/52.png" height="50" width="50">                  |  <img src="emojis/j7/2262.png" height="50" width="50"> <img src="emojis/j7/57.png" height="50" width="50"> <img src="emojis/j7/52.png" height="50" width="50"> <img src="emojis/j7/2321.png" height="50" width="50"> <img src="emojis/j7/2261.png" height="50" width="50">                  |
| <img src="imgs/COCO_train2014_000000001072.jpg" height="224" width="224"> | A plane is close to the ground near a mountain and trees |  <img src="emojis/i8/1393.png" height="50" width="50"> <img src="emojis/i8/715.png" height="50" width="50"> <img src="emojis/i8/716.png" height="50" width="50"> <img src="emojis/i8/63.png" height="50" width="50"> <img src="emojis/i8/2187.png" height="50" width="50">                  |  <img src="emojis/j8/716.png" height="50" width="50"> <img src="emojis/j8/718.png" height="50" width="50"> <img src="emojis/j8/1393.png" height="50" width="50"> <img src="emojis/j8/715.png" height="50" width="50"> <img src="emojis/j8/496.png" height="50" width="50">                  |
| <img src="imgs/COCO_train2014_000000001424.jpg" height="224" width="224"> | A picture of some family having a birthday party |  <img src="emojis/i9/2222.png" height="50" width="50"> <img src="emojis/i9/69.png" height="50" width="50"> <img src="emojis/i9/2234.png" height="50" width="50"> <img src="emojis/i9/44.png" height="50" width="50"> <img src="emojis/i9/2203.png" height="50" width="50">                  |   <img src="emojis/j9/2203.png" height="50" width="50"> <img src="emojis/j9/2226.png" height="50" width="50"> <img src="emojis/j9/2234.png" height="50" width="50"> <img src="emojis/j9/2238.png" height="50" width="50"> <img src="emojis/j9/2235.png" height="50" width="50">                 |
