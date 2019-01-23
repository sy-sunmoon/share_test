# Review Generator

## Key-words Based Review Generator
The model aims to generate fluent and reasonable reviews based on the input keywords describing the product.

***************************************************************

### Data Preprocess
Before running the preprocess.py, your should provide the following files in the <code>data/source_data/</code> folder:

- <code>XX.src1</code> is the file of the input keywords.
- <code>XX.src2</code> is the file of the concepts extracted from ConceptNet. 
- <code>XX.tgt</code> is the file of the output reviews.

Run preprocess.py as following, and the preprocessed files are stored in the <code>data/save_data/</code> folder.
```bash
python3 preprocess.py --load_data data/source_data/ --save_data data/save_data/
```

***************************************************************

### Train
To train a model, go to the review generation folder and run the following command:
```bash
python3 train.py --gpus gpu_id --config config.yaml --log log_name 
```

***************************************************************

### Test
To test the well-trained model, go to the review generation folder and run the following command:
```bash
python3 predict.py --gpus gpu_id --config config.yaml --restore checkpoint_path --log log_name 
```

***************************************************************

## Review Sentiment Transfer

The model learns to transfer a negative sentiment review into a positive one without any parallel data.

***************************************************************

### Data Preprocess
After running the preprocess/format_data.py, it can generate three files in the <code>sentiment_transfer</code> folder:

<code>train.0</code>, <code>dev.0</code>, <code>test.0</code> denotes the negative train/dev/test files

<code>train.1</code>, <code>dev.1</code>, <code>test.1</code> denotes the positive train/dev/test files
<br>

***************************************************************

### Train

To train a model, go to the sentiment-transfer folder and run the following command:
```bash
python style_transfer.py --train ../data/sentiment_transfer/train --dev ../data/sentiment_transfer/dev --output ../tmp/sentiment.dev --vocab ../tmp/google.vocab --model ../tmp/model
```

***************************************************************

### Test

#### Test file has sentiment labels
If the test file has sentiment labels, just run the following command:
```bash
python style_transfer.py --test ../data/sentiment_transfer/test --output ../tmp/sentiment_transfer.test --vocab ../tmp/google.vocab --model ../tmp/model --load_model true
```

#### Test file doesn't have sentiment labels
If the test file doesn't have sentiment labels, such as the generated reviews, just run the following model to train a binary sentiment classifier. And then load the trained model to detect which generated review is negative or positive.
```bash
# train
python classifier.py --train ../data/sentiment_transfer/train --dev ../data/sentiment_transfer/dev --vocab ../tmp/google.vocab --model ../tmp/classifer-model 
# test
python classifier.py --test TEST_FILE_PATH --output OUTPUT_FILE_PATH --vocab ../tmp/google.vocab --model ../tmp/model --load_model true
```
And then, run the follow code to get the transferred review:
```bash
python style_transfer.py --test OUTPUT_FILE_PATH --output ../tmp/sentiment_transfer.test --vocab ../tmp/google.vocab --model ../tmp/model --load_model true
```

<br>

***************************************************************

# Cite
This code is based on the following paper:

<i> "Style Transfer from Non-Parallel Text by Cross-Alignment". Tianxiao Shen, Tao Lei, Regina Barzilay, and Tommi Jaakkola. NIPS 2017. [arXiv](https://arxiv.org/abs/1705.09655) </i>

<i> "End-To-End Memory Networks". Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus. NIPS 2015. [arXiv](https://arxiv.org/abs/1503.08895) </i>

<i> "Dynamic Memory Networks for Visual and Textual Question Answering". Caiming Xiong, Stephen Merity, Richard Socher. 2017. [arXiv](https://arxiv.org/abs/1603.01417) </i>

## Dependencies
Python >= 2.7
