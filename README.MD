# Transcriberr
## Transcribe any video or audio file.

It uses OpenAI's Whisper to transcribe any text. The UI is made with Streamlit. 
Audio is automatically extracted from video files using ffmpeg. 

## Install
```bash
pip install git+https://github.com/Daankrol/whisper-streamlit.git#egg=whisper
```

## Running
Start the UI with: 
```python
python -m streamlit run main.py
```

NOTE: always use only one tab. If you use more than one, the model will also be loaded more than once. 