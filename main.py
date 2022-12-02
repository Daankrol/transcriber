import gradio as gr
import streamlit as st
import whisper
import os
import sys
from io import StringIO
from moviepy.editor import VideoFileClip

print("repaint")
# Streamlit rerurns this file everytime a user does an action, so we need to make sure that the model is only loaded once


def toggle_disable(state=None):
    if state is None:
        st.session_state.disable_transcribe_button = (
            not st.session_state.disable_transcribe_button
        )
    else:
        st.session_state.disable_transcribe_button = state


st.set_page_config(layout="wide")

# greet the user with a title
st.title("Transcriber")
# subtitle
st.write("Transcribe audio and video files with Whisper")
# add two columns beneath the container
col1, col2 = st.columns(2)
# Add a model selection widget with a default value of "medium" and choices as: tiny, base, small, medium, large
modelName = col1.selectbox(
    "Select model", ("tiny", "base", "small", "medium", "large"), index=3
)

# create a file uploader
file = col1.empty()
file = col1.file_uploader(
    "Upload a file",
    type=[
        "mp4",
        "wav",
        "mp3",
        "avi",
        "mov",
        "mkv",
        "mpg",
        "mpeg",
        "mp2",
    ],
)

placeholder_button = col1.empty()

statusMessageComponent = col1.empty()
# add a container beneath this file uploader
fileDownloadContainer = col1.container()
# add a container for the transcribed text
resultTextContainer = col2.container()


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
    audio.write_audiofile("audio.wav")
    # return absolute path to the audio file
    return os.path.abspath("audio.wav")


def process_video(file_str_path):
    print("processing file: ", file_str_path)
    if check_if_video(file_str_path):
        statusMessageComponent.text("File is a video, extracting audio from video...")
        file_str_path = extract_audio_from_video(file_str_path)
    try:
        print("going to transcribe file", file_str_path)
        statusMessageComponent.text(
            "Loading language model, this might take a while..."
        )
        modelInstance = whisper.load_model(modelName if modelName else "medium")
        statusMessageComponent.text("Transcribing file...")
        result = whisper.transcribe(
            modelInstance,
            file_str_path,
            verbose=True,
            streamlit_status_component=statusMessageComponent,
            streamlit_result_component=resultTextContainer,
        )
    except TypeError as e:
        print(e)
        statusMessageComponent.text("Are you sure that is a correct audio file?")
        return {"text": "Something went wrong", "segments": []}, True
    except Exception as e:
        print(e)
        statusMessageComponent.text("Something went wrong: \n" + str(e))
        return {"text": "Something went wrong", "segments": []}, True
    return result, False


if file:
    transcribe_button = placeholder_button.button(
        "Transcribe",
        key="but-transcribe2",
        disabled=False,
    )
    statusMessageComponent.text("File uploaded, ready to transcribe!")
    print("file name:", file.name)
    if transcribe_button:
        transcribe_button = placeholder_button.button("Transcribing...", disabled=True)
        # placeholder_button.button("Transcribing...", disabled=True)
        # write the file to disk
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        # get the absolute path to the file
        file_str_path = os.path.abspath(file.name)
        # process the file
        print("going to process file:", file_str_path)
        result, error_occured = process_video(file_str_path)
        print(result)
        if not error_occured:
            # save the file such that the user can download it again
            with open("result.txt", "w") as f:
                whisper.utils.write_txt(result["segments"], file=f)
            with open("result.srt", "w") as f:
                whisper.utils.write_srt(result["segments"], file=f)

            with open("result.txt", "r") as f:
                fileDownloadContainer.download_button(
                    label="Download text",
                    data=f,
                    file_name="result.txt",
                    mime="text/plain",
                )
            with open("result.srt", "r") as f:
                fileDownloadContainer.download_button(
                    label="Download srt",
                    data=f,
                    file_name="result.srt",
                    mime="text/plain",
                )

            # show the result as html such that we can enable scrolling
            col1.markdown(result["text"])
            # col1.markdown(
            # f"<div style='height: 300px; overflow-y: scroll;'>{result['text']}</div>",
            # )
