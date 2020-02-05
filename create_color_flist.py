import os
import argparse
import numpy as np

ext = {'.jpg', '.png', '.JPG'}
images = []
lr_path = '../train_color'
output = './train_color.flist'

for roots, dirs, files in os.walk(lr_path):
    
     for file in files:
        
         if os.path.splitext(file)[1] in ext:
             images.append(os.path.join(roots, file))
images = sorted(images)
np.savetxt(output, images, fmt='%s')

test_path = '../test_color'
output = './test_color'
images = []
for roots, dirs, _ in os.walk(test_path):
    for dir in dirs:
        for root, _, files in os.walk(os.path.join(roots, dir)):
            for file in files:
                if os.path.splitext(file)[1] in ext:
                    images.append(os.path.join(root, file))

images = sorted(images)
div4 = len(images) // 4
for i in range(4):
    np.savetxt(output + str(i) + '.flist', images[i*div4: (i+1)*div4], fmt='%s')

