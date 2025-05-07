regression.py:
Hypothesis A: There is a positive correlation between the distance from a 
political community and the negative sentiments of those communities.

Methodology & description: We hypothesize that there is a positive correlation between a community’s 
distance from politically affiliated subreddits and the level of negative sentiment within that community. 
To test this, we perform sentiment analysis on subreddit content to assign each post a qualitative 
sentiment score (ranging from negative to positive). These scores are then averaged at the community 
level to obtain an overall sentiment value per subreddit. By combining these sentiment averages with 
graph-based distances from a set of core political subreddits we constructed a dataset that allows
us to examine the relationship between proximity to political discourse and emotion. 
Cross-validation was also done, and our L2-norm value is 2.504820923622704.
We got a p_value of 0.0 <= 0.05. Hence, we can reject the null hypothesis that there is no correlation 
between the two variables as the low p_value indicates statistical significance for the correlation.
As we got a correlation slope of 0.0288, it suggests a  positive correlation between the distance from the root and positivity of the sentiment. 
As a result, we must reject our hypothesis as subreddits further from the root are actually more positive in sentiment. However, the R^2 value was 0.0054 which is very low and suggests that 
this correlation has little variance and is not necessarily the best predictive model for the relationship between the two variables. 


Challenges:
One challenge was accurately mapping subreddit relationships to compute meaningful distances from political communities.
Another was ensuring sentiment scores reliably reflected overall community tone despite noise and variability 
in the individual posts, as reflected in the low R-squared value of 0.0567.

Conclusions:
Based on the results we can see that there is a statistically significant 
positive correlation between subreddit distance from political communities 
and average sentiment. Communities further from political roots tend to be 
less negative or more emotionally neutral/positive. Hence, we reject the 
null hypothesis in favor of hypothesis A.


2_Sample_T_Test.py:

Research Null Hypothesis: There is no difference between the overall sentiment of Republican connected 
subreddits vs Democrat connected subreddits.

Methodology: We first utilized a two-sample T-test because we wanted to determine whether the population 
means for the sentiment of all Republican connected subreddits and Democrat connected subreddits were the 
same. We had considered using a Mann-Whitney U Test instead, but the sampling size, 568 Republican and 
220 Democrat, was high enough to utilize the Central Limit Theorem and assume Normality of the sampling 
distribution. Republican connected subreddits were connected to by breadth first search to 
r/conservative, r/republican, and r/liberal; democrat connected subreddits were connected to 
r/liberal, r/democrats, and r/progressive. Muddassar/longformer-base-sentiment-5-classes was 
used to measure sentiment. The top 40 most recent top posts were used in the sentiment calculation.

Interpretation: Since the p-value = 0.0067 < 0.01, we can reject the null hypothesis. Therefore, 
there is a difference between the overall sentiment of Republican connected subreddits and 
Democrat connected subreddits. These results corresponded with my initial estimates.
 We estimated that the different parties would have different sentiments due to the 
 difference in values and beliefs. We believe the tools for the analysis were accurate, 
 as 2-Sample T-Test is designed to compare 2 different population means to see if they are the same.

Challenges: We were not able to train our own sentiment model, as we didn’t have the data or the 
computational resources available. We tried using off-the-shelf sentiment models but many couldn't 
fit the context window for Reddit posts. After testing different models, we were finally able to
 get the open-source Muddassar/longformer-base-sentiment-5-classes model to accurately measure sentiment. 
 This model worked as the only one to work for our use case, but it only had a testing accuracy of 74.84%. 
 Additionally, there were some subreddit that were connected to both Republican and Democrat roots. 
 We chose the party that occurred first computationally, but this does mean that some of our data 
 doesn’t perfectly fit into one category or another.

 Clustering.py:

Hypothesis: There is a negative correlation between the distance between two nodes and how politcally biased they are. 

Method: We calculated the distance in terms of the number of nodes for each subreddit with respect to the seed nodes.

Then we took the text data for each subreddit and calculated its political bias. To do this, 
we used a transformer-based model from huggingface (a BERTbased model) called ("bucketresearch/politicalBiasBERT"). 
This essentially returned the percentage of how much left, center, right each text was. We used the ratio of left to right 
to get a political score. The lesser the score was the more right the text was and the higher the score/ratio was the more left the text was. 

Interpretation: We got a p_value of 0.034 <= 0.05. Hence, we can reject the null hypothesis.As we got a correlation slope of -1.6686 suggesting
a negative  coefficient. However, the R^2 value was 0.0019 which is very low which calls into question how statistically significant this correlation is. 

Challenges: The issue with the low R^2 value which indicates higher unexplained variance  may be partly because the BERT based model we used had a token limit. As a result we had to chunk the text piece 
by piece and then take the mean of the logits. This probably weakened the relevence and variance of the data.