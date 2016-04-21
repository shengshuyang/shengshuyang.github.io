---
layout: post
title:  "A step by step guide to Caffe!"
date:   2016-04-20 22:21:21 -0700
categories: Caffe
---

[Caffe](http://caffe.berkeleyvision.org/) is a great and very widely used framework for deep learning, offers a vast collection of out-of-the-box layers and an amazing ["model zoo"](https://github.com/BVLC/caffe/wiki/Model-Zoo), however, it's also famous for its lack of documentation.

After playing around with it for a few days, I felt that it would be great to share what I did to get everything up and running, especially some small hacks that I had to hunt around to get. I think it could well serve as a tutorial.

Before we get started, I strongly recommend going through [This Course](http://cs231n.stanford.edu/) to get a theoretical primer about convolutional neural networks. The course has a great tutorial on Caffe as well, although it's somewhat less detailed.

Now let's get started.

0. Get a desktop with a nice GPU!

Although most deep learning platforms can be run on CPU, it's in general much slower. My personal experience is around 50-100 times slower, and I timed it once: *160 images per 35 seconds* versus *5120 images in 26 seconds*. 

Up to this point I still think building a PC with a decent GPU is the best option (more flexible on IDE's, flexible time of computation etc.), but if you don't like to spend money on hardware, you can also use AWS or Terminal.com instead.

I found this great [**blog post**](http://timdettmers.com/2014/08/14/which-gpu-for-deep-learning/) on GPU selections. Take a look if you are wondering which GPU to buy.

1. Install Caffe

The official documentation of Caffe has a pretty detailed instruction on installing Caffe [**here**](http://caffe.berkeleyvision.org/installation.html); they also provided a few platform specific step by step tutorials (e.g. [**here**](http://caffe.berkeleyvision.org/install_apt.html)). Currently Ubuntu is best supported.

Thanks to the life saving *apt-get install* command, I was able to install most of the dependencies effortlessly. The Caffe package itself, however, needs to be compiled locally, some prior knowledge about *make* would help (not recommended to use the cmake script, it caused some issue for me).

2. Data Preparation

Caffe is a high performance computing framework, to get more out of its amazing GPU accelerated training, you certainly don't want to let file I/O slow you down, which is why a database system is normally used. The most common option is to use lmdb.

If you don't have experience with database systems, basically lmdb will be a huge file on your computer once you finished data preparation. You can query any file from it using a scripting language, and reading big chunks of data from it is much faster can reading a file.

Caffe has a tool `convert_imageset` to help you build lmdb from a set of images. Once you build your Caffe, the binary will be under `/build/tools`. There's also a bash script under `/caffe/examples/imagenet` that shows how to use `convert_imageset`.

You can also check out my recent [**post**](http://shengshuyang.github.io/hook-up-lmdb-with-caffe-in-python.html) on how to write images into lmdb using Python.

Either way, once you are done, you'll get two folders like this:

{% highlight bash %}
train_lmdb/
  -- data.mdb
  -- lock.mdb
val_lmdb/
  -- data.mdb
  -- lock.mdb
 {% endhighlight %}

one for training and one for validation. Under each folder, the `data.mdb` file will be huge, it's where your images actually goes in.

3. Setting up the model and the solver

Caffe has a very nice abstraction that separates neural network definitions (models) from the optimizers (solvers). A model defines the structure of a neural network, while a solver defines all information about how gradient descent will be conducted.

A typical model looks like this (note that the lmdb files we generated are specified here!):

{% highlight bash %}
# train_val.prototxt
name: "MyModel"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: "data/train_mean.binaryproto"
  }
  data_param {
    source: "data/train_lmdb"
    batch_size: 128
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: "data/train_mean.binaryproto"
  }
  data_param {
    source: "data/val_lmdb"
    batch_size: 50
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 0.01
    decay_mult: 1
  }
  param {
    lr_mult: 0.01
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
...
...
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "fc8"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "fc8"
  bottom: "label"
  top: "loss"
}
{% endhighlight %}

This is actually a part of the AlexNet, you can find its full definition under `/caffe/models/bvlc_alexnet`.

If you use Python, install `graphviz`, you can use a script `/caffe/python/draw_net.py` to visualize the structure of your network and check if you made any mistake in the specification.

![The BVLC Net]({{ site.url }}/images/bvlc_net.jpg)

