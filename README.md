# CogniScan AI

Explainable multimodal system for three-stage
cognitive decline detection from spontaneous speech.

## Setup

pip install -r requirements.txt
streamlit run app.py

## Usage

1. Upload a .cha transcript file
2. Optionally upload .mp3 audio
3. Enter CDS score if longitudinal data available
4. Click Analyse

## Model

RF+GB+SVM soft voting ensemble
Weighted F1 = 0.658 | Macro AUC = 0.850
Dataset: DementiaBank Pitt Corpus N=326