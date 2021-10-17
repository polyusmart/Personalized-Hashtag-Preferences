<!--
 * @Author: your name
 * @Date: 2021-10-17 00:21:46
 * @LastEditTime: 2021-10-17 14:50:28
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Personalized-Hashtag-Preferences/README.md
-->
# Personalized-Hashtag-Preferences

The official implementation of EMNLP 2021 paper "#HowYouTagTweets: Learning User Hashtagging Preferences via Personalized Topic Attention".

## Dataset

### Data statistics

| Number of Tweets                  | 33,881 |
|-----------------------------------|--------|
| Number of Users                   | 2,571  |
| Number of Hashtags                | 22,320 |
| Average tweet number per user     | 13     |
| Average hashtags number per user  | 12     |
| Average tweet number per hashtag  | 3      |
| New Hashtag Rate (%)              | 55     |

## Model

Our model that couples the effects of hashtag context encoding (top) and user history encoding (bottom left) with a personalized topic attention (bottom right) to predict user-hashtag engagements.

![](https://raw.githubusercontent.com/Yb-Z/images/main/20211017144829.png)

## Code

### Dependencies


### Preprocess

### Pretrain

### Train

### Evaluation

We use [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) to evaluate the predictions in the file ```sortTest.dat```, which is in the format requirements of <mark>RankLib</mark>, run: 

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