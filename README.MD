# Transcriberr
## Transcribe any video or audio file.

It uses OpenAI's Whisper to transcribe any text. The UI is made with Streamlit. 
Audio is automatically extracted from video files using ffmpeg. 

## Install
```bash
conda create -n transcriberr python=3.10
pip install -r requirements.txt
brew install ffmpeg

#pip install git+https://github.com/Daankrol/whisper-streamlit.git#egg=whisper
```

## Running
Start the UI with: 
```bash
python -m streamlit run main.py
```

NOTE: always use only one tab. If you use more than one, the model will also be loaded more than once. 


# WhisperX
```bash
conda create --name whisperx python=3.10
conda activate whisperx
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install git+https://github.com/m-bain/whisperx.git
```

Install ffmpeg and rust:
```bash
brew install ffmpeg
```

Usage: 
```bash
whisperx <audio_file> --model large-v2 --diarize --highlight_words True
```