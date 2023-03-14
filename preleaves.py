from matplotlib import image
from matplotlib import pyplot
import pandas as pd
import os
import numpy as np
import csv

dpath = __file__.replace('\\', '/')[:__file__.rfind('\\')+1] + 'data/leaves/'

test = image.imread(dpath+'leaves/Gauva (P3)/healthy/0004_0001.JPG')
# summarize shape of the pixel array
print(test.dtype)
print(test.shape)
# display the array of pixels as an image
# pyplot.imshow(image)
# pyplot.show()

file = open(dpath+'train.csv', "w", newline="")
writer = csv.writer(file)

for dir in os.listdir(dpath+'/leaves'):
    for ddir in os.listdir(dpath+'/leaves/'+dir):
        for fpath in os.listdir(dpath+'/leaves/'+dir+'/'+ddir):
            label = fpath[0:4]+fpath[5:9]
            label = int(label)
            im = np.array(image.imread(
                dpath+'/leaves/'+dir+'/'+ddir+'/'+fpath))
            im = im.flatten()
            print(im.shape)
            writer.writerow(np.insert(im, 0, label))

file.close()
