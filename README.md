

https://user-images.githubusercontent.com/80037791/181944422-b70a3b1a-61fb-4d04-8091-4f7dc14409b1.mp4

# Emotion Predictoin App

Hi! I am Rakesh Sahni B.E Final year computer science.
Current website : https://sohipm.com


process to build this app
1.) First stage is undoubtly data acquisition -> https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?select=train.txt

2.) In NLP data cleaning and data munging is one of the crucial part we have to remove punctuation, stop words, html tag. spell correct etc. To do so i have used spacy nlp library

3.) After for model building we use 2 approch. First is machine learning and 2nd is deep learning means LSTM recurrent neural network.

4.) During model building I relise there is Imblance data set among
  joy         5362, 
  sadness     4666, 
  anger       2159, 
  fear        1937, 
  love        1304, 
  surprise     572


When I used machine learning then we just remove love and surprise categories due to less percentage data in whole corpus. I have also another method for over-sampling or under sampling or we can also use SMOT but I did'nt use here.
Accuracy with about 80-85%

when I use LSTM recurent neural network in that situation. I did'nt remove lesser categories because
I have use advance pretrained glove word embedding. ( stanford university )
Accuracy with about 85-92%

## Installation

open terminal and type simply

```bash
pip install -r requirements.txt
```

After installation all python package simply write command in terminal

```bash
python app.py
```
Remember you are in app ( root ) folder

#### Thank you!
