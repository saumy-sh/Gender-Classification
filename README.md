# CNN based Gender classification
Trained a CNN based model on [dataset](https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal) using mel-spectrogram features.
Here is a simple UI of this
<img width="1515" height="382" alt="ui" src="https://github.com/user-attachments/assets/90ce5e41-4e2d-46f1-b56c-2801d4a98dd8" />
Here you have the option to load the audio as well as record one on the fly and get the predictions after clicking Submit.
## Training Results
Here is a plot showing train and validation accuracy as well as loss.
![accuracy-loss](https://github.com/user-attachments/assets/27b3d929-9b79-4842-874b-fc40c7d8dd45)
## Running the repo
```bash
git clone https://github.com/saumy-sh/Gender-Classification
```
Once cloned, run the command to install required dependencies
```bash
pip install -r requirements.txt
```
To run the gradio app, run following command:
```bash
python app.py
```
Here you will one localhost link and sharable public link. You can use anyone by clicking on it.
