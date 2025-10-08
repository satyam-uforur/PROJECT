### **Audio Dataset for Agricultural Queries**

Overview



This dataset contains synthetic audio samples of common agricultural queries in Hindi, generated using the gTTS library. Each query is recorded in both female and male voice variants. The dataset is intended for building voice-based applications such as intent recognition for farmers.



**Dataset Structure**

dataset/

│

├── female\_001.wav

├── male\_001.wav

├── female\_002.wav

├── male\_002.wav

│   ...

├── dataset.csv







WAV files: Audio recordings of the queries.



female\_XXX.wav – Female voice.



male\_XXX.wav – Male voice.



dataset.csv: Metadata file containing mapping of audio files to their corresponding text and intent.



**CSV Format**



The CSV file (dataset.csv) has the following columns:



file\_name	                     text	                   intent

female\_001.wav	Mausam thoda thanda hai ya garam bataiye	weather\_info

male\_001.wav	Mausam thoda thanda hai ya garam bataiye	weather\_info

...	...	...



file\_name: Name of the audio file.



text: The original text of the query.



intent: The intent category of the query, such as:



weather\_info

fertilizer\_use

pest\_control

crop\_recommendation

irrigation\_advice









#### **Voice-to-Intent Classification Pipeline**

Overview



This project implements a voice-to-intent classification system for agricultural queries. It takes audio recordings, converts them to text using OpenAI Whisper, and classifies the intent (such as weather information, crop suggestions, or fertilizer queries) using TF-IDF features and a Logistic Regression classifier.



The main steps of the pipeline are:

1.Preprocess audio (convert to mono and resample to 16 kHz).

2.Transcribe audio into text using Whisper.

3.Convert text into numerical features using TF-IDF.

4.Train a Logistic Regression classifier for intent prediction.



**here is notebook**=*https://www.kaggle.com/code/satyamtiwari09/ai-task-1*



**Folder Structure**



project/

│

├─ main.py       # fast api logic

├─ index.html        # frontend

├─ tfidf\_intent\_model.pkl            # Trained Logistic Regression model

├─ tfidf\_vectorizer.pkl              # Trained TF-IDF vectorizer

├─ Requirements.txt





**Audio Preprocessing**



The audio files are first loaded while keeping their original sampling rate. If they are stereo, they are converted to mono. All files are resampled to 16 kHz, which is the recommended sample rate for Whisper. The processed files are saved in the folder audio\_dataset\_preprocessed.



This ensures consistent input for transcription and reduces computational complexity.



**Transcription with Whisper**



Each audio file is passed through Whisper to convert speech to text. The resulting transcriptions are stored in a CSV file named whisper\_transcriptions.csv. This file contains the file name and the corresponding transcribed text.



**TF-IDF and Logistic Regression for Intent Classification**



The transcribed text is used as input features, and the intent labels are used as targets. Text is converted into numerical vectors using TF-IDF, which captures the importance of words in the dataset. Logistic Regression is then trained on these features to classify the intent of each query.



The trained model and vectorizer are saved as tfidf\_intent\_model.pkl and tfidf\_vectorizer.pkl respectively for later use.



**Prediction on New Audio**



To predict the intent of a new audio file, first transcribe it using Whisper. Then, transform the text using the saved TF-IDF vectorizer and use the Logistic Regression model to predict the intent.



**Why This Pipeline?**



Audio preprocessing: Converts audio to mono and 16 kHz to ensure compatibility with Whisper.



Whisper ASR: Converts spoken queries into text for further processing.



TF-IDF Vectorizer: Converts text into numerical features suitable for machine learning. The output dimension is based on the number of features selected (e.g., 5000).



Logistic Regression: Efficient multi-class classifier for intent prediction.



This pipeline ensures that spoken queries can be accurately transcribed and classified with minimal preprocessing and computational overhead.



References

Librosa: Audio processing library.

OpenAI Whisper: Speech-to-text model.

Scikit-learn: TF-IDF feature extraction and Logistic Regression.





#### **INSTALLATIONS**

cd project
(ensure u are in project folder)

**Create and activate virtual environment**



python -m venv venv

\# Activate

\# Windows: venv\\Scripts\\activate

\# Linux/Mac: source venv/bin/activate





**Install dependencies**



pip install -r requirements.txt



**start server**



uvicorn main:app --reload --port 8000











