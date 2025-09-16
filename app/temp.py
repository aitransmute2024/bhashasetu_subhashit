import os
import shutil
import tempfile
import subprocess
from app.routes.pipeline import complete_pipeline
from moviepy import VideoFileClip, AudioFileClip

def process_video_with_subtitles(input_video_path: str, target_language: str) -> str:
    """
    Processes a video: runs the pipeline, replaces audio, adds subtitles, 
    and returns the final video path. Uses temporary files for intermediate steps.

    Args:
        input_video_path (str): Path to the original video.
        target_language (str): Target language for audio and subtitles.

    Returns:
        str: Path to the final processed video.
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"❌ Input video file not found: {input_video_path}")

    # Create temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
    shutil.copy(input_video_path, temp_video_path)

    try:
        # Step 1: Run the pipeline
        final_audio, final_srt = complete_pipeline(temp_video_path, target_language)
        if not final_audio or not os.path.exists(final_audio):
            raise FileNotFoundError(f"❌ Final audio file not generated: {final_audio}")
        if not final_srt or not os.path.exists(final_srt):
            raise FileNotFoundError(f"❌ Final SRT file not generated: {final_srt}")

        # Step 2: Replace audio
        final_video_path = os.path.join(temp_dir, "final_video.mp4")
        video = VideoFileClip(temp_video_path)
        new_audio = AudioFileClip(final_audio)
        video_with_audio = video.set_audio(new_audio)
        video_with_audio.write_videofile(final_video_path, codec="libx264", audio_codec="aac")

        # Step 3: Add subtitles
        output_video_path = os.path.join(os.getcwd(), f"final_{os.path.basename(input_video_path)}")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", final_video_path,
            "-vf", f"subtitles={final_srt}",
            "-c:a", "copy",
            output_video_path
        ]
        subprocess.run(cmd, check=True)

        return output_video_path

    except Exception as e:
        raise RuntimeError(f"❌ Error processing video: {str(e)}")
    
    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


# Example usage:
# if __name__ == "__main__":
#     input_video = r"C:\Users\Sidhant Raj\Desktop\Sidhant\bhashasetu_subhashit\samples\second_sample.mp4"
#     target_lang = "hindi"
#     final_video = process_video_with_subtitles(input_video, target_lang)
#     print(f"🎉 Final processed video saved at: {final_video}")
