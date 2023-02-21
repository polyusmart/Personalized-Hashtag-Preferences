<!--
 * @Author: your name
 * @Date: 2021-10-17 00:21:46
 * @LastEditTime: 2021-10-17 19:51:10
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Personalized-Hashtag-Preferences/README.md
-->

# Personalized-Hashtag-Preferences

The official implementation of EMNLP 2021 paper "[#HowYouTagTweets: Learning User Hashtagging Preferences via Personalized Topic Attention](https://aclanthology.org/2021.emnlp-main.616/)". Here we give some representative commands illustrating how to preprocess data, train, test, and evaluate our model. The code for topic model session is mainly adapted from YueWang’s TAKG code. Thanks for Yue’s support!

## Dataset

This Twitter dataset was first gathered with the official streaming API in Feb 2013, which contains 900M tweets. Our twitter dataset can be found at `Data` directory.

### Data format

- From the raw data:

  1. We filtered out tweets without hashtags and capped the user history at 50 tweets.
  2. Hashtag texts are hidden from both history and contexts to avoid the trivial features learned by the models, and the tweets presenting hashtags in the middle were ignored to enable better semantic learning (following Wang et al. (2019)).
  3. We removed users who posted original hashtags only (never tagged by others) because these users cannot be taken for prediction.

- For training and evaluation, we rank the tweets
  by time and take the earliest 80% for training, the
  latest 10% for test, and the remaining 10% for validation.
- For each segment (train/valid/test), each line is a tweet with tweet id, several hashtags, separated by '\t'.

### Data statistics

<table class="tg">
<tbody>
  <tr>
    <td>Number of Tweets</td>
    <td>33,881</td>
  </tr>
  <tr>
    <td>Number of Users </td>
    <td>2,571</td>
  </tr>
  <tr>
    <td>Number of Hashtags</td>
    <td>22,320</td>
  </tr>
  <tr>
    <td>Average tweet number per user </td>
    <td>13</td>
  </tr>
  <tr>
    <td>Average hashtags number per user </td>
    <td>12</td>
  </tr>
  <tr>
    <td>Average tweet number per hashtag </td>
    <td>3</td>
  </tr>
  <tr>
    <td>New Hashtag Rate (%) </td>
    <td>55</td>
  </tr>
</tbody>
</table>

## Model

Our model that couples the effects of hashtag context encoding (top) and user history encoding (bottom left) with a personalized topic attention (bottom right) to predict user-hashtag engagements.

<img src="https://raw.githubusercontent.com/Yb-Z/images/main/20211017144829.png" width="380"/>

## Code

### Dependencies

- Python 3.8+
- Pytorch 1.7.1

### Topic Data Preprocess

> If you do not need to change the data, you can skip the Topic Data Preprocessing and Neural Topic Model Pretrain sessions.(The processed and pretrained data is saved in the `Data`, `Processed_data` and `NTMData` directory. Go to training session to run the `prediction.py` directly.)

The code of TAKG is slightly changed to suit the model. The topic data for input is processed as the format of Yue’s input Data(cite and show the data format).

> If you want to regenerate it, create a scratch_dataset object in `utils/scratch_dataset.py`, the getvaefile and writevaefile functions will regenerate the input data.

For preprocess, turn to the TAKG directory, run:

```bash
python preprocess.py -data_dir data/StackExchange
```

### Neural Topic Model Pretrain

This step is to get the pretrained NTM embeddings(only train ntm for 20 epochs) for later training and testing.
Pretrain the neural topic model, turn to the TAKG directory, run:

```bash
python train.py -data_tag StackExchange_s150_t10 -only_train_ntm -ntm_warm_up_epochs 20
```

### Training and Testing

> If you use the existing preprocessed and pretrained data, skip this data-move step. If you changed and regenerate the preprocessed and pretrained data, move the preprocessed data(TAKG/processed_data) to the main directory(processed_data), move the pretrained data(TAKG/data/StackExchange) to the main directory(NTMData).

First the other parameters except ntm are warmed up for 20 epochs, then the all parameters are updated for 100 epochs buy joint training.

For training and testing, turn to the main directory, run:

```bash
python predict.py
```

### Evaluation

To generate the evaluation file, first turn to the main directory, run:

```bash
python sortBypredict.py
```

which will read two files in the `Records` directory (i.e., test2.dat and pre.txt) and output the evaluation file (i.e., `sortTest.dat`).

Then, use [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) to evaluate the predictions in `sortTest.dat`, which is in the format requirements of <mark>RankLib</mark>, run:

```bash
java -jar RankLib.jar -test sortTest.dat -metric2T <metric> -idv <file>
```

\<metric>: Metric to evaluate on the test data. We used <mark>MAP</mark>, <mark>P@5</mark>, <mark>nDCG@5</mark> in our paper.

\<file>: The output file to print model performance (in test metric) on individual ranked lists (has to be used with -test).

The output example is the following:

```
Discard orig. features
Model file:
Feature normalization: No
Test metric: <metric>
Reading feature file [sortTest.dat]... [Done.]
( ranked lists,  entries read)
<metric> on test data: <result>
Per-ranked list performance saved to: <file>
```

## Please cite our work if you use this code and dataset:

```
@inproceedings{zhang-etal-2021-howyoutagtweets,
    title = "{\#}{H}ow{Y}ou{T}ag{T}weets: Learning User Hashtagging Preferences via Personalized Topic Attention",
    author = "Zhang, Yuji  and
      Zhang, Yubo  and
      Xu, Chunpu  and
      Li, Jing  and
      Jiang, Ziyan  and
      Peng, Baolin",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.616",
    pages = "7811--7820",
    abstract = "Millions of hashtags are created on social media every day to cross-refer messages concerning similar topics. To help people find the topics they want to discuss, this paper characterizes a user{'}s hashtagging preferences via predicting how likely they will post with a hashtag. It is hypothesized that one{'}s interests in a hashtag are related with what they said before (user history) and the existing posts present the hashtag (hashtag contexts). These factors are married in the deep semantic space built with a pre-trained BERT and a neural topic model via multitask learning. In this way, user interests learned from the past can be customized to match future hashtags, which is beyond the capability of existing methods assuming unchanged hashtag semantics. Furthermore, we propose a novel personalized topic attention to capture salient contents to personalize hashtag contexts. Experiments on a large-scale Twitter dataset show that our model significantly outperforms the state-of-the-art recommendation approach without exploiting latent topics.",
}


Thanks to the follower's attention, we found that the version of code is not our SOTA version. We were regret that due to the crash of our server in 2021, we encountered  the management disorder of versions of our code. This existing version is not our SOTA version. Nevertheless, if you want to follow our work, we still encourage you to check several sections of our code for reference: the data processing pipeline and the model network. The dataset is well-processed and well-organized with auto-annotation. If you adopt the dataset, please cite our work. Additionally, the effectiveness of our SOTA model (especially the module of personalized topic attention mechanism) was tested for multiple times with abundant ablation experiments. We encourage you to implement the work from the beginning and make fair comparison between the Lstm-Att (past SOTA) and our SOTA model. If there is any question, welcome to discuss with us by email. In the meanwhile, we are continuously trying to get the code versions in order.
