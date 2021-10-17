<!--
 * @Author: your name
 * @Date: 2021-10-17 00:21:46
 * @LastEditTime: 2021-10-17 15:01:31
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Personalized-Hashtag-Preferences/README.md
-->

# Personalized-Hashtag-Preferences

The official implementation of EMNLP 2021 paper "#HowYouTagTweets: Learning User Hashtagging Preferences via Personalized Topic Attention". Here we give some representative commands illustrating how to preprocess data, train, test, and evaluate our model. The code for topic model session is mainly adapted from YueWang’s TAKG code. Thanks for Yue’s support!


## Dataset

### Data format

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

![](https://raw.githubusercontent.com/Yb-Z/images/main/20211017144829.png)

## Code

### Dependencies

### Topic Data Preprocess

PS: If you do not need to change the data, you can skip the Topic Data Preprocessing and Neural Topic Model Pretrain sessions.(The processed and pretrained data is saved in the ‘Data’, ‘Processed_data’ and ‘NTMData’ directory. Go to training session to run the prediction.py directly.)

The code of TAKG is slightly changed to suit the model. The topic data for input is processed as the format of Yue’s input Data(cite and show the data format). 
PS: If you want to regenerate it, create a scratch_dataset object in utils/scratch_dataset.py, the getvaefile and  writevaefile functions will regenerate the input data.

For preprocess, turn to the TAKG directory, run:
Python preprocess.py -data_dir data/StackExchange


### Neural Topic Model Pretrain

This step is to get the pretrained NTM embeddings(only train ntm for 20 epochs) for later training and testing. 
Pretrain the neural topic model, turn to the TAKG directory, run:
Python train.py -data_tag StackExchange_s150_t10 -only_train_ntm -ntm_warm_up_epochs 20


### Training and Testing

PS:If you use the existing preprocessed and pretrained data, skip this data-move step.
If you changed and regenerate the preprocessed and pretrained data, move the preprocessed data(TAKG/processed_data) to the main directory(processed_data), move the pretrained data(TAKG/data/StackExchange) to the main directory(NTMData). 

First the other parameters except ntm are warmed up for 20 epochs, then the all parameters are updated for 100 epochs buy joint training. 
 
For training and testing, turn to the main directory, run:
Python predict.py


### Evaluation

We use [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) to evaluate the predictions in the file `sortTest.dat`, which is in the format requirements of <mark>RankLib</mark>, run:

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

## Citation
