# Emotion Detection in Conversations

---

## Requirements

- python3.7
- pytorch
- torchnlp

---

## Model

The main idea is from paper [DialogueRNN An Attentive RNN for Emotion Detection in Conversations](https://arxiv.org/pdf/1811.00405.pdf).

And there are two main parts of this model

1. Feature extraction

   Used BiLSTM to get one sentence's vector.

2. Emotion Detection

   Mainly composed of three GRUCell.

---

## Dataset

`DailyDialogue`

---

## Config

In this project pretrained word vectors `GloVe` was used. And in `utils.py` you can adjust the store path `TORCHNLP_CACHEDIR` of vectors file.

---

## Usage

- train

  `$ python3 main.py --istrain`

- test

  `$ python3 main.py`

- other options

  `$ python3 main.py --help`

---

*If you think this project is helpful to you, plz star it and let more people to see it. :)*
