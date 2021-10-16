<!--
 * @Author: your name
 * @Date: 2021-10-17 00:21:46
 * @LastEditTime: 2021-10-17 00:57:23
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Personalized-Hashtag-Preferences/README.md
-->
# Personalized-Hashtag-Preferences

The official implementation of EMNLP 2021 paper "#HowYouTagTweets: Learning User Hashtagging Preferences via Personalized Topic Attention".

### Evaluation

We use [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) to evaluate the predictions.

To evaluate the predictions in the file ```sortTest.dat```, which is already meet the format requirements of <mark>RankLib</mark> for test data .
run: 

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