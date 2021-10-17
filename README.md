<!--
 * @Author: your name
 * @Date: 2021-10-17 00:21:46
 * @LastEditTime: 2021-10-17 15:01:31
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Personalized-Hashtag-Preferences/README.md
-->

# Personalized-Hashtag-Preferences

The official implementation of EMNLP 2021 paper "#HowYouTagTweets: Learning User Hashtagging Preferences via Personalized Topic Attention".

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

### Preprocess

### Pretrain

### Train

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
