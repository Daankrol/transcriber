from moviepy.editor import VideoFileClip
import os

def check_if_video(filename):
    # if file ends with .mp4, .mpg, .mov, .mkv, .avi, .flv, .wmv, .webm, .mpeg, .m4v, .3gp, .3g2, .f4v, .f4p, .f4a, .f4b
    if filename.lower().endswith(
        (
            ".mp4",
            ".mpg",
            ".mov",
            ".mkv",
            ".avi",
            ".flv",
            ".wmv",
            ".webm",
            ".mpeg",
            ".m4v",
            ".3gp",
            ".3g2",
            ".f4v",
            ".f4p",
            ".f4a",
            ".f4b",
        )
    ):
        return True

def extract_audio_from_video(filename):
    # use moviepy to extract the audio from the video
    video = VideoFileClip(filename)
    audio = video.audio
    # to results folder
    audio.write_audiofile("results/audio.wav")
    # return absolute path to the audio file
    return os.path.abspath("results/audio.wav")
