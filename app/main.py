import os
from routes.pipeline import complete_pipeline
from moviepy import VideoFileClip, AudioFileClip

video_path = "C:/Users/admin/OneDrive - Aidwise Private Ltd/BhashaSetu_VAM/samples/input.mp4"
target_language = "tamil"

# Step 1: Check if video file exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"❌ Input video file not found: {video_path}")

# Step 2: Run the pipeline with error handling
try:
    final_audio = complete_pipeline(video_path, target_language)
    if not final_audio or not os.path.exists(final_audio):
        raise FileNotFoundError(f"❌ Final audio file not generated or missing: {final_audio}")
except Exception as e:
    raise RuntimeError(f"❌ Error during pipeline execution: {str(e)}")


def replace_audio(video_path, new_audio_path, output_path):
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ Video file not found: {video_path}")
        if not os.path.exists(new_audio_path):
            raise FileNotFoundError(f"❌ New audio file not found: {new_audio_path}")

        print("✅ Loading video and new audio...")
        video = VideoFileClip(video_path)
        new_audio = AudioFileClip(new_audio_path)

        print("✅ Replacing audio...")
        final_video = video.set_audio(new_audio)

        print(f"✅ Writing final video to: {output_path}")
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print("🎉 Final video created successfully.")

    except Exception as e:
        raise RuntimeError(f"❌ Error while replacing audio: {str(e)}")


# Example usage
replace_audio(video_path, final_audio, "final_video.mp4")
