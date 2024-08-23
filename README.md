# phish-links-detection

This repo implements a solution for detecting phishing links using artificial neural network's multi-layer perceptron

## Getting started
git clone https://github.com/BensonsPage/ml_based_phish-links-detection

cd phish-links-detection

Ensure you have python 3.8 ++
## Install PIP if Not Installed

python3 -m pip install / 

python3 -m pip install --upgrade pip

## Create the Virtual Environment

python3 -m venv venv

## start virtual environment

source venv/bin/activate

## Install Requirements

pip install -r requirements.txt

## Fitting/Training the model

python3 dnn_model.py

Close data visiualization charts as they get generated to proceed to next step.

## Classifying a new link as phish/benign

python3 detect.py

To infer a single URL, you need to first convert the features into the right format as highlighted on detect.py
