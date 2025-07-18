from routes.pipeline import complete_pipeline
from moviepy import VideoFileClip, AudioFileClip

video_path = "samples/sample.mp4"
target_language = "hindi"
final_audio = complete_pipeline(video_path, target_language)



def replace_audio(video_path, new_audio_path, output_path):
    # Load video and audio
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(new_audio_path)

    # Set new audio to video
    final_video = video.set_audio(new_audio)

    # Export final video
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Example usage
replace_audio(video_path, final_audio, "final_video.mp4")
