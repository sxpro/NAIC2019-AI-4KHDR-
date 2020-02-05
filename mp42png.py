from subprocess import call
import subprocess
import os
import os.path as osp
from multiprocessing.dummy import Pool as ThreadPool
import cv2

def error_frame(name): #问题帧
    error_frame_name = ['10099858/068.png', '10099858/081.png', '10099858/089.png', '10099858/091.png',
                        '15922480/001.png', '15922480/006.png', '31545121/045.png', '32490669/001.png',
                        '38001368/047.png', '78821035/030.png', '97231911/055.png']
    _name = name.split('/')[-2] + '/' + name.split('/')[-1]
    if _name in error_frame_name:
        return True
    else:
        return False

def get_frame_types(video_fn): #get key_frame
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

def save_i_keyframes(video_fn):
    video_hr = osp.join(sourcedir_hr, osp.basename(video_fn))
    save_path_lr = osp.join(submit_lr, osp.basename(video_fn))[:-4]
    save_path_hr = osp.join(submit_hr, osp.basename(video_fn))[:-4]
    os.makedirs(save_path_hr, exist_ok=True)
    os.makedirs(save_path_lr, exist_ok=True)
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    if i_frames:
        cap = cv2.VideoCapture(video_fn)
        cap_hr = cv2.VideoCapture(video_hr)
        for frame_no in i_frames:
            str_frame_no = str(frame_no+1)
            lr_name = save_path_lr + '/' + str_frame_no.zfill(3) + '.png'
            hr_name = save_path_hr + '/' + str_frame_no.zfill(3) + '.png'

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            cap_hr.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            ret, frame_hr = cap_hr.read()
            cv2.imwrite(lr_name, frame)
            cv2.imwrite(hr_name, frame_hr)
        cap.release()
        cap_hr.release()
    print ('success in '+osp.basename(video_fn))


def process2img(folder):
    save_path = osp.join(submit_test, osp.basename(folder))[:-4]
    os.makedirs(save_path, exist_ok=True)

    os.system('ffmpeg -i {} -vsync 0  -f image2 {}/%02d.png'.format(folder, save_path))
if __name__ == '__main__':
	
    sourcedir_lr = '../train_2nd/540p' #540p视频存放位置
    sourcedir_hr = '../train_2nd/4K' #4K视频存放位置
    sourcedir_test = '../test' #test ideo
    submit_lr = '../lr_images'  #540p视频的关键帧存放位置
    submit_hr = '../hr_images'  #4K视频的关键帧存放位置
    submit_test = '../test_data'
    os.makedirs(submit_lr, exist_ok=True)
    os.makedirs(submit_hr, exist_ok=True)
    os.makedirs(submit_test, exist_ok=True)

    folders = [osp.join(sourcedir_lr, f) for f in os.listdir(sourcedir_lr)]
    test_folders = [osp.join(sourcedir_test, f) for f in os.listdir(sourcedir_test)]
    pool = ThreadPool()
    pool.map(process2img, test_folders) #先生成测试集， 如有错误请分开执行
    pool.map(save_i_keyframes, folders) #再生成训练集
    pool.close()
    pool.join()
