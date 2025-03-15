import os
import numpy as np
import h5py
import cv2
import imageio
import argparse
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

def video_frame_sample(frame_interval, video_length, sample_num):
    num = []
    for l in range(video_length):
        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))
    return num

def extract_features(video_dir, output_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    
    lis = os.listdir(video_dir)
    len_data = len(lis)
    video_features = np.zeros([len_data, 10, 7, 7, 512])
    
    t = 10  # length of video
    sample_num = 16  # frame number for each second
    
    for num in range(len_data):
        video_index = os.path.join(video_dir, lis[num])
        vid = imageio.get_reader(video_index, 'ffmpeg')
        
        metadata = vid.get_meta_data()
        fps = metadata['fps']
        duration = metadata['duration']
        vid_len = int(fps * duration)
        frame_interval = int(vid_len / t)
        frame_num = video_frame_sample(frame_interval, t, sample_num)
        
        imgs = [cv2.resize(im, (224, 224)) for im in vid]
        vid.close()
        
        extract_frame = [imgs[n] for n in frame_num]
        feature = np.zeros(([10, 16, 7, 7, 512]))
        
        for j, y_im in enumerate(extract_frame):
            x = img_to_array(y_im)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            pool_features = np.float32(model.predict(x))
            
            tt = int(j / sample_num)
            video_id = j - tt * sample_num
            feature[tt, video_id, :, :, :] = pool_features
        
        feature_vector = np.mean(feature, axis=1)
        video_features[num, :, :, :, :] = feature_vector
    
    with h5py.File('./video_cnn_feature.h5', 'w') as hf:
        hf.create_dataset("dataset", data=video_features)
    
    print(f"Feature extraction complete. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CNN features from video frames.")
    parser.add_argument("video_dir", type=str, help="Path to the directory containing video files")
    args = parser.parse_args()
    extract_features(args.video_dir)
