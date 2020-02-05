    
from subprocess import call
import subprocess
import os
import os.path as osp
from multiprocessing.dummy import Pool as ThreadPool
import cv2
# call(["ffmpeg", "-i", filename, "-r", "5", pict])  # 这里的5为5fps，帧率可修改
# ffmpeg -i 16536366.mp4 -vf select='eq(pict_type\,I)' -vsync 2  -f image2 ..\\test\\%02d.png
# ffmpeg -r 24000/1001 -i pngs/out%4d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 test.mp4

def process2video(folder):
    save_path = osp.join(submit, osp.basename(folder))
    # os.makedirs(save_path, exist_ok=True)
    os.system('ffmpeg -r 24000/1001 -i '+ folder + '/%3d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 '+ save_path +'.mp4 -y')
    # print('1')
def process2img(folder):
    save_path = osp.join(submit, osp.basename(folder))[:-4]
    os.makedirs(save_path, exist_ok=True)
    
    os.system('ffmpeg -i {} -vsync 0  -f image2 {}/%02d.png'.format(folder, save_path))


def get_frame_types(video_fn):
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            cap_hr.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            ret, frame_hr = cap_hr.read()
            lr_name = save_path_lr + '/' + str(frame_no) + '.png'
            hr_name = save_path_hr + '/' + str(frame_no) + '.png'
            cv2.imwrite(lr_name, frame)
            cv2.imwrite(hr_name, frame_hr)
        cap.release()
        cap_hr.release()
    print ('success in '+osp.basename(video_fn))


if __name__ == '__main__':
    #### change here
    sourcedir = '/media/vipsl2018/93fc6e00-21de-45b4-b2e0-918253481277/xps/LDR_540p'
    submit = '/media/vipsl2018/93fc6e00-21de-45b4-b2e0-918253481277/xps/train_lr'
    os.makedirs(submit, exist_ok=True)
    ####
		
	
    #sourcedir_lr = '/mnt/xps/4KSR/data/LDR_540p'
    #sourcedir_hr = '/mnt/xps/4KSR/data/SDR_4K'
    #submit_lr = '/home/xps/4KSR/lr_images'
    #submit_hr = '/home/xps/4KSR/hr_images'
    #os.makedirs(submit_lr, exist_ok=True)
    #os.makedirs(submit_hr, exist_ok=True)
	

    folders = [osp.join(sourcedir, f) for f in os.listdir(sourcedir)]
    # for folder in folders:
    #     save_i_keyframes(folder)
    pool = ThreadPool(3)
    pool.map(process2img, folders)
    pool.close()
    pool.join()
