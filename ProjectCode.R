# Loading required packages

library(tm)
library(dplyr)
library(tidytext)
library(xlsx)
library(tidyr)
library(caret)
library(RWeka)
library(e1071)

# Defining functions to be used for data preprocessing and extracting features

to.ascii <- function(str){
  ## This function converts text from UTF-8 encoding to ASCII encoding,
  ## while replacing any unrecognizable characters with a blank
  sapply(str,iconv, from ="UTF-8", to="ASCII", sub="")
}

corpus.generate <- function(str){
  ## This function reads in text and converts it into a vector corpus. This makes it easier to
  ## perform further pre-processing. This particular function does the following pre-processing:
  ## 1. Transform text to lower case
  ## 2. Remove numbers
  ## 3. Remove punctuation
  ## 4. Remove stopwords like and, the, or etc.
  corpus <- Corpus(VectorSource(str))
  corpus <- corpus %>%
            tm_map(content_transformer(tolower)) %>%
            tm_map(removeNumbers) %>% 
            tm_map(removePunctuation) %>% 
            tm_map(removeWords, stopwords("English"))
  corpus
}

dtm.generate <- function(corpus, ng){
  ## This function takes in a corpus and the number of ngrams to be created and creates the 
  ## required number of n-grams. The function only creates the specified number of n-grams and 
  ## ignores the subset of n-grams. For example, ng=2 will create 2-grams but will not create
  ## 1-grams
  BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ng, max = ng))
  tempdtm <- DocumentTermMatrix(corpus, control=list(tokenize=BigramTokenizer)) 
  tempdtm
}

class.out <- function(str_pred, str_actual){
  ## This function calculates the accuracy and creates a confusion matrix using the predicted and
  ## the actual labels
  ClassMatrix = table(str_pred, str_actual)
  Accuracy = sum(diag(ClassMatrix))/sum(ClassMatrix)
  list(Mat=ClassMatrix, Accuracy=Accuracy)
}

# Reading the dataset in a variable 'raw'
raw <- read.csv("fake_or_real_news_cleaned.csv", stringsAsFactors = FALSE)
names(raw)<-c("X","title","text","label")

# Converting title and text to ASCII format and replacing missing text values with 'noentryentered'
raw <- raw %>% 
       mutate(DocID=seq_len(nrow(raw)),text=to.ascii(text), title=to.ascii(title), label=as.factor(label))%>%
       mutate_at(vars(text),funs(replace(., is.na(.), "noentryentered")))

# Storing each of the columns into three different columns
raw_title <- raw %>% select(DocID, title)
raw_text <- raw %>% select(DocID, text)
raw_label <- raw %>% select(DocID, FakeorRealLabel=label)

set.seed(12345)
# Partitioning data into training and test set
inTrain<-createDataPartition(y=raw$label,p=0.75,list = FALSE)

# Genearate pre-processed corpus for title and text
corpus_title <- corpus.generate(raw$title) 
corpus_text <- corpus.generate(raw$text)

# Create 1-grams for title and text
n <- 1 
dtm_title <- dtm.generate(corpus_title, n) 
dtm_text <- dtm.generate(corpus_text,n)

# Set sparsity index
sparse <- 0.995 
df_title <- dtm.to.df(dtm_title, sparse, "title")
df_text <- dtm.to.df(dtm_text, sparse, "text")

# Create the final dataframe which can be used for model building
data <- raw_label %>% 
        inner_join(df_text, by="DocID") %>% 
        inner_join(df_title, by="DocID") %>% 
        select(-DocID)

train <- data[inTrain,]
test <- data[-inTrain,]

# Create a Naive Bayes model
model <- naiveBayes(FakeorRealLabel~., data=train)

# Predict on the test set
pred <- predict(model, test, type="class")

# View the confusion matrix and the accuracy results
out <- class.out(pred, test$FakeorRealLabel)

out$ClassMatrix
out$Accuracy
