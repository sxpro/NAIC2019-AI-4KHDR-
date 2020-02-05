from subprocess import call
import subprocess
import os
import os.path as osp
from multiprocessing.dummy import Pool as ThreadPool
import cv2
def process2video(folder):
    save_path = osp.join(submit, osp.basename(folder))
    # os.makedirs(save_path, exist_ok=True)
    os.system('ffmpeg -r 24000/1001 -i ' + folder + '/%2d.png -vcodec libx265 -pix_fmt yuv422p -crf 6 ' + save_path +'.mp4 -y')
    # print('1')
if __name__ == '__main__':
    #### change here
    sourcedir = 'Results'
    submit = '../answer'
    os.makedirs(submit, exist_ok=True)
    ####

    folders = [osp.join(sourcedir, f) for f in os.listdir(sourcedir)]

    pool = ThreadPool()
    pool.map(process2video, folders)
    pool.close()
    pool.join()
