import streamlit as st
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


def translate_eng_to_dutch(text, tokenizer, model):
    # translate english to dutch

    # tokenize the text
    tokenized_text = tokenizer(text, return_tensors="pt")
    # translate the text
    translated = model.generate(**tokenized_text)
    # decode the text
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text[0]


st.set_page_config(layout="wide")

# live text state variable
if "live_text" not in st.session_state:
    st.session_state.live_text = ""

# greet the user with a title
st.title("Transcriber")
# subtitle
st.write(
    "Transcribe and diarize audio and video files using Whisper, Wav2Vec, Nvidia NeMo, Facebook Demucs, Voice Activity Detection and Speaker Diarization"
)
# create a file uploader
upc1, upc2 = st.columns(2)
file = upc1.empty()
file = upc1.file_uploader(
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


placeholder_button = upc2.empty()

statusMessageComponent = st.empty()
# create results directory if it does not exist
if not os.path.exists("results"):
    os.makedirs("results")

col1, col2 = st.columns(2)
fileDownloadContainer = col1.container()
# translate button container


st.markdown(
    """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """,
    unsafe_allow_html=True,
)
resultCol1, resultCol2 = st.columns(2)
# add small header to each column
resultCol1.subheader("Transcription")
resultCol2.subheader("Translation")
# add a container for the transcribed text
# translate result download button
translate_button_container = col2.empty()
resultTextContainer = resultCol1.container()
print(st.session_state.live_text)
resultTextContainer.markdown(st.session_state.live_text)
translateTextContainer = resultCol2.container()


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

        # save result in state variable
        st.session_state.result = result
        st.session_state.error_occured = error_occured
        st.session_state.isDoneTranscribing = True

        if not error_occured:
            # save the file such that the user can download it again
            # use the file name as the name of the file

            srtName = file.name.split(".")[0] + ".srt"
            srtName = os.path.join("results", srtName)
            # with open(srtName, "w") as f:
            #     whisper.utils.write_srt(result["segments"], file=f)
            txtName = file.name.split(".")[0] + ".txt"
            txtName = os.path.join("results", txtName)
            # with open(txtName, "w") as f:
            # whisper.utils.write_txt(result["segments"], file=f)

            # create a zip file
            zipName = "transcribed_" + file.name.split(".")[0] + ".zip"
            zipName = os.path.join(zipName)

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

# if the user has transcribed a file, show a button for starting a translation
# check if key present
if "isDoneTranscribing" in st.session_state and st.session_state.isDoneTranscribing:
    if not st.session_state.error_occured:
        translate_button = translate_button_container.button(
            "Translate to Dutch",
        )
        if translate_button:
            # get the result from the state variable
            result = st.session_state.result
            # get the text from the result
            text = result["text"]

            # show message
            statusMessageComponent.text("Loading translation model...")
            # load the translation model
            # from eng to dutch
            # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
            # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")

            # translate the text
            statusMessageComponent.text("Translating...")
            # translation = translate_eng_to_dutch(text, tokenizer, model)
            translation = "hoi"
            statusMessageComponent.text("Done translating!")
            # show the translation in markdown so we can auto line break
            translateTextContainer.markdown(translation)
            # save the translation in results
            translation_name = "translation_" + file.name.split(".")[0] + ".txt"
            with open("results/translation.txt", "w") as f:
                f.write(translation)
            # download button for the translation
            with open("translation.txt", "w") as f:
                f.write(translation)
            with open("translation.txt", "rb") as f:
                fdb = fileDownloadContainer.download_button(
                    "Download translation",
                    data=f,
                    file_name=translation_name,
                    mime="text/plain",
                )
                # if clicked, remove all files from the results folder
                if fdb:
                    for file in os.listdir("results"):
                        os.remove(os.path.join("results", file))
                    os.remove("translation.txt")
                    print("removed all files from results folder")
