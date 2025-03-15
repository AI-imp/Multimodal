import argparse
import imageio
from moviepy.editor import VideoFileClip

def process_video(input_path, output_path, target_fps, target_duration):
    # 打开原始视频
    input_video = imageio.get_reader(input_path, 'ffmpeg')
    metadata = input_video.get_meta_data()
    original_fps = metadata['fps']
    duration = metadata['duration']
    num_frames = int(original_fps * duration)
    
    target_num_frames = int(target_fps * target_duration)
    frame_interval = max(1, int(original_fps / target_fps))
    
    output_video_path = f"{output_path}_video.mp4"
    output_audio_path = f"{output_path}_audio.wav"
    final_output_path = f"{output_path}_final.mp4"
    
    output_video = imageio.get_writer(output_video_path, fps=target_fps)
    for i in range(target_num_frames):
        frame_index = i * frame_interval
        if frame_index < num_frames:
            frame = input_video.get_data(frame_index)
            output_video.append_data(frame)
    
    input_video.close()
    output_video.close()
    
    video_clip = VideoFileClip(input_path)
    audio_clip = video_clip.audio.subclip(0, target_duration)
    audio_clip.write_audiofile(output_audio_path, codec='pcm_s16le')
    
    output_video_clip = VideoFileClip(output_video_path).set_audio(audio_clip)
    output_video_clip.write_videofile(final_output_path, codec='libx264', audio_codec='aac')
    
    audio_clip.close()
    video_clip.close()
    output_video_clip.close()
    
    print(f"处理完成，最终视频保存至: {final_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频裁剪与重采样")
    parser.add_argument("input", type=str, help="输入视频文件路径")
    parser.add_argument("output", type=str, help="输出文件名前缀")
    parser.add_argument("--fps", type=float, default=25.0, help="目标帧率 (默认 25.0)")
    parser.add_argument("--duration", type=float, default=10.05, help="目标时长 (默认 10.05s)")
    args = parser.parse_args()
    process_video(args.input, args.output, args.fps, args.duration)