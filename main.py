import streamlit as st
import whisper
import os
from moviepy.editor import VideoFileClip
import zipfile


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
# create results directory if it does not exist
if not os.path.exists("results"):
    os.makedirs("results")

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
    # to results folder
    audio.write_audiofile("results/audio.wav")
    # return absolute path to the audio file
    return os.path.abspath("results/audio.wav")


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
            # use the file name as the name of the file

            srtName = file.name.split(".")[0] + ".srt"
            srtName = os.path.join("results", srtName)
            with open(srtName, "w") as f:
                whisper.utils.write_srt(result["segments"], file=f)
            txtName = file.name.split(".")[0] + ".txt"
            txtName = os.path.join("results", txtName)
            with open(txtName, "w") as f:
                whisper.utils.write_txt(result["segments"], file=f)

            # create a zip file
            zipName = "transcribed_" + file.name.split(".")[0] + ".zip"
            zipName = os.path.join(zipName)
            print("zipName", zipName)

            with zipfile.ZipFile(zipName, "w") as zip:
                zip.write(srtName)
                zip.write(txtName)

            with open(zipName, "rb") as f:
                fdb = fileDownloadContainer.download_button(
                    "Download transcribed file",
                    data=f,
                    file_name=zipName,
                    mime="application/zip",
                )
                # if clicked, remove all files from the results folder
                if fdb:
                    for file in os.listdir("results"):
                        os.remove(os.path.join("results", file))
                    os.remove(zipName)
                    print("removed all files from results folder")
