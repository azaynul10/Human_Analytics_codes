from moviepy.editor import VideoFileClip
import os

# Configure paths
input_dir = r"c:\Users\Abedi\Downloads\OneDrive_2025-01-31\not fall"
output_dir = r"c:\Users\Abedi\OneDrive - Student Ambassadors\archive (7)\audio_extrator_of_not_fall_videos"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Process all video files
for video_file in os.listdir(input_dir):
    if video_file.endswith((".mp4", ".avi", ".mov")):
        video_path = os.path.join(input_dir, video_file)
        audio_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.wav")
        
        try:
            # Extract audio
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(audio_path, codec='pcm_s16le')  # WAV format for ML models
            video.close()
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
