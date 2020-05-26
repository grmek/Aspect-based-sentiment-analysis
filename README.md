# Aspect-based sentiment analysis
Authors: Miha Bizjak, Anže Gregorc, Rok Grmek

## What is this?
This is a project for the Natural language processing course (FRI UL 2019/2020).
Our task is to get the subjective information from text material that refer to a entity with the use of natural language processing and other methods.
A entity is considered as a person, organization or a location and can be represented multiple times in one document or a sentence and there could be more entities in one document.
All methods are evaluated on the [SentiCoref 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1285) data set.

## Prerequisites
$ pip3 install -r requirements.txt  
$ python3  
&gt;&gt;&gt; import nltk  
&gt;&gt;&gt; nltk.download('punkt')

## Trained models
Training of tested models takes a lot of time.
Trained models are available for download [here](https://www.sendtransfer.com/download.php?id=f82802d706da70a694f974b9da74ae31&email=1498741).
Copy the downloaded models in the 'data/models' directory.

## How to run it?
$ cd src  
$ python3 main.py

## Results
All results produced by this source code are available in the 'results.txt' file.
