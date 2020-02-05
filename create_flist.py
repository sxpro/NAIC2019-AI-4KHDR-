import os
import argparse
import numpy as np

ext = {'.jpg', '.png', '.JPG'}
lr_path = '../lr_images'
output = './train_lr.flist'

images = []

for roots, dirs, files in os.walk(lr_path):
    
    for file in files:
        
        if os.path.splitext(file)[1] in ext:
            images.append(os.path.join(roots, file))
images = sorted(images)
np.savetxt(output, images, fmt='%s')

hr_path = '../hr_images'
output = './train_hr.flist'

images = []

for roots, dirs, files in os.walk(hr_path):
    
    for file in files:
        
        if os.path.splitext(file)[1] in ext:
            images.append(os.path.join(roots, file))
images = sorted(images)
np.savetxt(output, images, fmt='%s')

test_path = '../test_data'
output = './test.flist'
images = []
for roots, dirs, _ in os.walk(test_path):
    for dir in dirs:
        for root, _, files in os.walk(os.path.join(roots, dir)):
            for file in files:
                if os.path.splitext(file)[1] in ext:
                    images.append(os.path.join(root, file))

images = sorted(images)
np.savetxt(output, images, fmt='%s')
