import streamlit as st
import os

import zipfile
from types import SimpleNamespace
import time
from utils import check_if_video, extract_audio_from_video
from convert_output import convert

from transformers import pipeline

import sentry_sdk

sentry_sdk.init(
    dsn="https://47fe40328a7d72860ac879ce9a7cc372@o4506060209192960.ingest.sentry.io/4506847038013440",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)


if not os.path.exists("results"):
    os.makedirs("results")


def translate_eng_to_dutch(text):
    # check if translation pipeline is in st state if not create it
    if "translate_pipe" not in st.session_state:
        st.session_state.translate_pipe = pipeline(
            "translation", model="Helsinki-NLP/opus-mt-en-nl"
        )
    translation = st.session_state.translate_pipe(text)
    return translation[0]["translation_text"]


st.set_page_config(layout="wide")
st.title("Transcriber")
st.write(
    "Transcribe and diarize audio and video files using Whisper  \n  \n\
Whisper-Large-V3 is the largest and most accurate model, but also the slowest  \n\
Distil-Whisper is a smaller and faster model, but can be a bit less accurate  \n\
Whisper-Tiny is the smallest and fastest model, use only for debugging  \n  \n\
If you have out-of-memory issues, try reducing the batch size."
)

# Initialize button states
if "transcribe_button_enabled" not in st.session_state:
    st.session_state.transcribe_button_enabled = False

if "translate_button_enabled" not in st.session_state:
    st.session_state.translate_button_enabled = False

## show parameter selections but now in two columsn:
col1, col2 = st.columns(2)
parameters = SimpleNamespace(
    model_name=col1.selectbox(
        "Model",
        options=[
            "openai/whisper-large-v3",
            "openai/whisper-tiny",
            "distil-whisper/distil-large-v2",
        ],
        index=0,
    ),
    device=col1.selectbox("Device", options=["mps", "cpu"], index=0),
    torch_dtype=col2.selectbox("Torch Dtype", options=["float32", "float16"], index=1),
    batch_size=col2.slider("Batch Size", min_value=1, max_value=32, value=4),
    task=col1.selectbox(
        "Transcribe any language or transcribe non-English audio to English?",
        options=[
            "transcribe",
            "translate",
        ],
        index=0,
    ),
    # translate_to_dutch=col2.checkbox(
    #     'Translate to Dutch (only possible if using "transcribe")', value=False
    # ),
)

# divider:
st.markdown(
    """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """,
    unsafe_allow_html=True,
)

# create a file uploader
upc1, upc2 = st.columns(2)
file = upc1.empty()
file = st.file_uploader(
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
transcribe_button_placeholder, translate_button_container = st.columns(2)


statusMessageComponent = st.empty()
col1, col2 = st.columns(2)
fileDownloadContainer = col1.container()

translate_button_container = col2.container()
translateTextContainer = st.container()


def process_video(file_str_path):
    print("processing file: ", file_str_path)
    if check_if_video(file_str_path):
        statusMessageComponent.text("File is a video, extracting audio from video...")
        file_str_path = extract_audio_from_video(file_str_path)
        statusMessageComponent.empty()
    try:
        print("going to transcribe file", file_str_path)
        parameters.audio = file_str_path
        with st.spinner("Transcribing..."):
            result = st.session_state.pipe(
                parameters.audio,
                chunk_length_s=30,
                batch_size=parameters.batch_size,
                return_timestamps=True,
            )
        convert(result, "srt", "results", verbose=True)
        convert(result, "txt", "results", verbose=True)

    except TypeError as e:
        print(e)
        statusMessageComponent.text("Are you sure that is a correct audio file?")
        return {"text": "Something went wrong", "segments": []}, True
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(e)
        statusMessageComponent.text("Something went wrong: \n" + str(e))
        return {"text": "Something went wrong", "segments": []}, True

    return result, False


if file:
    # place the spinner in the placeholder:
    if "pipe" not in st.session_state:
        with st.spinner("Loading model, this could take a while..."):
            st.session_state.pipe = pipeline(
                "automatic-speech-recognition",
                model=parameters.model_name,
                device=parameters.device,
                batch_size=parameters.batch_size,
                torch_dtype=parameters.torch_dtype,
            )

    statusMessageComponent.text("Ready to transcribe!")
    # enable the transcribe button
    st.session_state.transcribe_button_enabled = True

    if transcribe_button_placeholder.button(
        "Transcribe",
        disabled=not st.session_state.transcribe_button_enabled,
        key="transcribe_button_disabled",
    ):
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        # get the absolute path to the file
        file_str_path = os.path.abspath(file.name)

        now = time.time()
        result, error_occured = process_video(file_str_path)
        duration = time.time() - now
        # save result in state variable
        st.session_state.result = result
        st.session_state.error_occured = error_occured
        st.session_state.isDoneTranscribing = True

        # show result in status message
        statusMessageComponent.text(result["text"])
        statusMessageComponent.text(f"Transcription took {duration:.2f} seconds")

        if not error_occured:
            # file is saved as results/output.srt, lets change the name to the original file name
            srtName = "results/" + file.name.split(".")[0] + ".srt"
            os.rename("results/output.srt", srtName)
            # create a txt file
            txtName = "results/" + file.name.split(".")[0] + ".txt"
            os.rename("results/output.txt", txtName)

            # # create a zip file
            zipName = "results/" + file.name.split(".")[0] + ".zip"

            st.session_state.file_names = [srtName, txtName, zipName]

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

# if the user has transcribed a file, show a button for starting a translation
# check if key present
if "isDoneTranscribing" in st.session_state and st.session_state.isDoneTranscribing:
    if not st.session_state.error_occured:
        st.session_state.translate_button_enabled = True

        if translate_button_container.button(
            "Translate to Dutch",
            disabled=not st.session_state.translate_button_enabled,
            key="translate_button_disabled",
        ):
            result = st.session_state.result
            text = result["text"]

            # show message
            statusMessageComponent.text("Loading translation model...")

            # translate the text
            statusMessageComponent.text("Translating...")
            start = time.time()
            translation = translate_eng_to_dutch(text)
            translation_duration = time.time() - start
            statusMessageComponent.text(
                f"Done translating, took {str(translation_duration)} seconds."
            )
            # show the translation in markdown so we can auto line break
            translateTextContainer.markdown(translation)
            translation_name = "results/" + file.name.split(".")[0] + "_translation.txt"
            # save the translation in results and save it as translation.txt
            with open(translation_name, "w") as f:
                f.write(translation)

            [srtName, txtName, zipName] = st.session_state.file_names
            zipName = "results/" + file.name.split(".")[0] + "_translation.zip"
            with zipfile.ZipFile(zipName, "w") as zip:
                zip.write(srtName)
                zip.write(txtName)
                zip.write(translation_name)

            # download button for the translation
            with open(translation_name, "rb") as f:
                fdb = fileDownloadContainer.download_button(
                    "Download translation",
                    data=f,
                    file_name=zipName,
                    mime="application/zip",
                )
                # if clicked, remove all files from the results folder
                if fdb:
                    for file in os.listdir("results"):
                        os.remove(os.path.join("results", file))
                    os.remove("translation.txt")
                    print("removed all files from results folder")
