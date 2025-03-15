# Multimodal audiovisual event localization
## 1.Convert the video  sampling
```
python3 video_process.py input_video.mp4 output_video --fps num1 --duration num2
```
## 2.Video feature extraction
```
python3 visual_feature_extractor.py yourvideo_dir
```
## 3.Train
```
python3 cmm_train.py yourdatapaths
```
## 4.Test
cmm_test.py
