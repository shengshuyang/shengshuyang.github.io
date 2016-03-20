---
layout: post
title:  "How to Hook Up lmdb with Caffe using Python!"
date:   2016-03-19 00:45:21 -0700
categories: Caffe
---

Long story short, here's how I figured out how to interact with lmdb using Python.

First, a bit of setup:

{% highlight python %}
import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
{% endhighlight %}

You've probably noticed two unfamiliar packages `caffe.proto` and `caffe.io`. 

The `caffe.proto` package defined a lot of things, but here we are only using the data structure `Datum`, you can think of it as an intermediate form between our images and labels and lmdb entries.

The `caffe.io` package has this two helper functions `datum = datum_to_array(X, y)` and `X = array_to_datum(datum)`, which could save us some time defining the structure of our `Datum` object. Note that `y` is not returned by `array_to_datum`, you can simply call `datum.label` to get it.

---------------------------

Function to write to lmdb:

{% highlight python %}
def write_images_to_lmdb(img_dir, db_name):
    for root, dirs, files in os.walk(img_dir, topdown = False):
        if root != img_dir:
            continue
        map_size = 64*64*3*2*len(files)
        env = lmdb.Environment(db_name, map_size=map_size)
        txn = env.begin(write=True,buffers=True)
        for idx, name in enumerate(files):
            X = mp.imread(os.path.join(root, name))
            y = 1
            datum = array_to_datum(X,y)
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())   
    txn.commit()
    env.close()
    print " ".join(["Writing to", db_name, "done!"])
{% endhighlight %}

This function takes a folder path `img_dir` as input and push all the images in the folder into a lmdb database specified by `db_name`.

`map_size` is the capacity of the database. In my case I have total of `len(files)` image patches of size `64*64*3`, and I used a factor of 2 just in case.

I've also set `y=1` for all images because currently I don't have labels yet. You will need to make changes according to your situation.

---------------------------

Function to read from lmdb:

{% highlight python %}
def read_images_from_lmdb(db_name, visualize):
	env = lmdb.open(db_name)
	txn = env.begin()
	cursor = txn.cursor()
	X = []
	y = []
	idxs = []
	for idx, (key, value) in enumerate(cursor):
		datum = caffe_pb2.Datum()
		datum.ParseFromString(value)
		X.append(np.array(datum_to_array(datum)))
		y.append(datum.label)
		idxs.append(idx)
	if visualize:
	    print "Visualizing a few images..."
	    for i in range(9):
	        img = X[i]
	        plt.subplot(3,3,i+1)
	        plt.imshow(img)
	        plt.title(y[i])
	        plt.axis('off')
	    plt.show()
	print " ".join(["Reading from", db_name, "done!"])
	return X, y, idxs
{% endhighlight %}

Here we iterate over all entries in the database, and visualize a few images if necessary. It is a good way to check if the images are processed correctly.
