## Data Source

The data has been sourced from the github repository of George McIntire which can be found at [here](https://github.com/GeorgeMcIntire/fake_real_news_dataset). The dataset contains 6335 observations and 3 columns:

1. News title
2. News text
3. Label signifying whether the news article is 'Fake' or 'Real'.

The dataset is balanced as 3164 observations are labeled as 'Fake' while 3171 observations are labeled as 'Real'.

## Data Preprocessing

### Missing Data

25 out of the 6335 observations do not contain any value in the “label” column, and these data are not included in further analysis.

Among the remaining 6310 observations, every observation has a title. However, several news articles do not contain any text in the main body and are certainly labeled as 'Fake'. The text entry of these observations is manually set to be *noentryentered* to treat these missing values appropriately.

### Text Preparation

Numbers, common words (such as the, a, etc.), and punctuation was removed as they frequently occur in almost all news articles and do not provide any significant information while building a model. Moreover, all text was converted to lower case to avoid discriminating between capitalized and lower case words.

### Document Term Matrix

The document term matrix (DTM) is a sparse matrix describing a collection of documents with one row for each document and one column for each term. This project considered only one word or two words at a time as a term. The value in each cell is then count of each term in each document. 

The columns of the document term matrix would be predictors for the classification model.

## Data Exploration

### Most Frequent Words/Phrases

From the DTM, we could identify words that appear most frequently among fake or real news (Exhibit 1~4).
The most frequent words and phrases that appeared in the titles of fake news include fbi, russia, world war and onion americas, which all seem quite irrelevant from real news. Meanwhile, the most frequent words and phrases that appeared in the titles of real news include debate, (bernie) sanders and ted bush, which have more details than fake news and are therefore more believable.

The same applies to the text bodies of the news, where fake news only frequency in general terms like hillary and trump, while real news involve more details like fox news.

In general, there exist certain differences in word frequencies between fake news and real news, and thus we could proceed to use them as predictors.

### Sentiment Analysis

The sentiment score of each news article reflects the general attitude of the news text. The higher the score, the more positive attitude in the news text.

The distribution of the sentiment score for fake news and real news respectively are shown in Figure 1. Fake news are abnormally condensed in a neutral narrative (sentiment score is zero) with some outliers, while real news are less condensed with a slight left skewness. This reflects the fact that news are usually written from a neutral perspective, and whether we could use the sentiment score as a predictor need to be tested in modelling.

## Training the model

The following approaches were implemented to come up with the best classification model:

1. **Building unigram vs bigram models:** Unigram models are those which consider each word as a separate predictor variable to build the model. Bigram models consider two words at a time, in addition to single words, and can help to detect signal from pairs of words that occur frequently. However, bigram models presented in this report use only pair of words to build the model, without including single words. The same can be extended to n words at a time, but it was not done in our analysis.

2. **Changing sparsity of the Document Term Matrix:** A matrix is said to be sparse when most of its values are zero. By reducing the sparsity of the matrix, terms with low frequencies can be removed, leading to a smaller matrix without affecting the performance of the model.

3. **Considering only the article text, ignoring the title text:** The rationale behind this was to see the impact on model performance by not considering the title text as predictors. If the model performance is almost the same, it would be preferable to have a model with lesser predictors.

4. **Not converting text to lowercase:** R’s text mining algorithms are case sensitive. This implies that uppercase words are treated differently from lowercase words. This might affect the model as some words like US (United States) would become the same as the word 'us' if every word is converted to lowercase.

5. **Data Mining Algorithms:** Due to the large size of term matrices, only a limited number of algorithms were implementable. Naive Bayes was chosen as the algorithm to be implemented, as it could be trained within a reasonable amount of time.

With these approaches in mind, the following models were constructed:

1. *Naive Bayes on a unigram DTM with sparsity of 99.5%*: 
The document term matrix for this model treated each word as a separate predictor. The sparsity of the matrix was 99.5% implying that words with a frequency lesser than 0.5% are removed from the DTM before the model is constructed. The model resulted in an accuracy of **83.31%** on the validation set.

2. *Naive Bayes on a unigram DTM with sparsity of 95%*: 
This model resulted in an accuracy of **78.17%**, which is lower than that for a DTM with 99.5% sparsity.

3. *Naive Bayes on a unigram DTM with sparsity of 90%*: 
This model resulted in an accuracy of **75.50%**. The accuracy further drops down.

4. *Naive Bayes on a unigram DTM consisting only of article text with sparsity of 99.5%*: 
A DTM consisting of only article text was considered while making the model. The resulting accuracy was **82.49%**. The accuracy is comparable to the original model, and it might be better to use this model compared to the first model as it uses a lesser number of predictors without a loss in accuracy.

5. *Naive Bayes on a unigram DTM without changing the case of the text, sparsity of 99.5%*: 
This model resulted in an accuracy of **83.44%**, which is comparable to the first model.

6. *Naive Bayes on a bigram DTM consisting only of article text with a sparsity of 99.5%*: 
This model resulted in an accuracy of **77.86%**. The accuracy might be low as there might be some signal present in individually considered words. A combination of unigram and bigram predictors could not be used due to lack of adequate computing resources.

7. *Naive Bayes on a bigram DTM consisting only of article text without changing case of the text, with a sparsity of 99.5%*: 
This model resulted in an accuracy of **79.63%**, which is slightly better than the previous model. This suggests that there might be some signal present in capitalized words as these are generally proper nouns, and their inclusion might improve the model performance significantly.

## Discussion and Conclusion
Based on the data mining algorithms that were implemented, the following observations were made:

1. The model accuracy deteriorated significantly as the sparsity was reduced
2. Unigram only models performed better than bigram only models. Combining the two models might result in a better model performance
3. Removing the title text did not impact the performance of the model significantly
4. Leaving the case of the words unchanged improved the model accuracy slightly
If a model were to be selected, we would choose a Naive Bayes model built on a DTM without changing the case of the text, with a sparsity of 99.5% as it results in the highest accuracy.

However, we feel that the model accuracy is not good. The model definitely performs better than a naive model, which has an accuracy of 50%, but there are improvements that can be made, especially considering the fact that the creator of the dataset could achieve an accuracy of around 91% with a Naive Bayes model.

Some possible areas for improvement are:

1. As discussed before, one possible approach would be to use a unigram model in conjunction with a bigram model. Improved accuracy might be achieved by using phrases made up of more than two words.

2. We calculated a sentiment score for each news article, but the same could not be included in the models. Building a model with a sentiment score might help extract more signal from the data.
