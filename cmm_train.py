from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Input, Lambda, Dense, LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
import h5py
import math
import random
random.seed(3344)

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def load_h5_data(filepath, dataset_name):
    with h5py.File(filepath, 'r') as hf:
        return np.array(hf[dataset_name][:], dtype=np.float32)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 2.0
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def scmm_net(input_audio_shape, input_video_shape):
    video_input = Input(shape=(input_video_shape,))
    video = Dense(128)(video_input)
    video = LeakyReLU(alpha=0.3)(video)
    video = Dense(64)(video)

    audio_input = Input(shape=(input_audio_shape,))
    audio = Dense(128)(audio_input)
    audio = LeakyReLU(alpha=0.3)(audio)
    audio = Dense(64)(audio)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([video, audio])
    return Model([video_input, audio_input], distance)

def main(args):
    closs_labels = load_h5_data(args.labels_path, 'avadataset')
    video_features = load_h5_data(args.video_features_path, 'avadataset')
    audio_features = load_h5_data(args.audio_features_path, 'avadataset')
    train_l = load_h5_data(args.train_order_path, 'order')
    val_l = load_h5_data(args.val_order_path, 'order')
    test_l = load_h5_data(args.test_order_path, 'order')

    x_audio_train = np.zeros((len(train_l) * 10, 128))
    x_video_train = np.zeros((len(train_l) * 10, 512))
    x_audio_val = np.zeros((len(val_l) * 10, 128))
    x_video_val = np.zeros((len(val_l) * 10, 512))
    x_audio_test = np.zeros((len(test_l) * 10, 128))
    x_video_test = np.zeros((len(test_l) * 10, 512))
    y_train = np.zeros((len(train_l) * 10))
    y_val = np.zeros((len(val_l) * 10))
    y_test = np.zeros((len(test_l) * 10))

    for i, id in enumerate(train_l):
        for j in range(10):
            x_audio_train[10 * i + j, :] = audio_features[id, j, :]
            x_video_train[10 * i + j, :] = video_features[id, j, :]
            y_train[10 * i + j] = closs_labels[id, j]

    for i, id in enumerate(val_l):
        for j in range(10):
            x_audio_val[10 * i + j, :] = audio_features[id, j, :]
            x_video_val[10 * i + j, :] = video_features[id, j, :]
            y_val[10 * i + j] = closs_labels[id, j]

    print("Data loading finished!")

    model = scmm_net(x_audio_train.shape[1], x_video_train.shape[1])
    model.compile(loss=contrastive_loss, optimizer=Adam())
    model.fit([x_video_train, x_audio_train], y_train, batch_size=8, epochs=20, validation_data=([x_video_val, x_audio_val], y_val))

    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, 'cmm_model.json'), 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(os.path.join(args.model_dir, 'cmm_model_weights.h5'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", type=str, required=True,default='data/labels_closs.h5')
    parser.add_argument("--video_features_path", type=str, required=True,default='data/visual_feature_vec.h5')
    parser.add_argument("--audio_features_path", type=str, required=True,default='data/audio_feature.h5')
    parser.add_argument("--train_order_path", type=str, required=True,default='data/train_order_match.h5')
    parser.add_argument("--val_order_path", type=str, required=True,default='data/val_order_match.h5')
    parser.add_argument("--test_order_path", type=str, required=True,default='data/test_order_match.h5')
    parser.add_argument("--model_dir", type=str, default="model")
    args = parser.parse_args()
    main(args)
