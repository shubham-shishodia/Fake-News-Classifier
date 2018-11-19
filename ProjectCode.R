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
  BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ng, max = ng))
  tempdtm <- DocumentTermMatrix(corpus, control=list(tokenize=BigramTokenizer)) 
  tempdtm
}

class.out <- function(str_pred, str_actual){
  ClassMatrix = table(str_pred, str_actual)
  Accuracy = sum(diag(ClassMatrix))/sum(ClassMatrix)
  list(Mat=ClassMatrix, Accuracy=Accuracy)
}

raw <- read.csv("fake_or_real_news_cleaned.csv", stringsAsFactors = FALSE)
names(raw)<-c("X","title","text","label")

raw <- raw %>% 
       mutate(DocID=seq_len(nrow(raw)),text=to.ascii(text), title=to.ascii(title), label=as.factor(label))%>%
       mutate_at(vars(text),funs(replace(., is.na(.), "noentryentered")))

raw_title <- raw %>% select(DocID, title)
raw_text <- raw %>% select(DocID, text)
raw_label <- raw %>% select(DocID, FakeorRealLabel=label)

set.seed(12345) 
inTrain<-createDataPartition(y=raw$label,p=0.75,list = FALSE)

corpus_title <- corpus.generate(raw$title) 
corpus_text <- corpus.generate(raw$text)

n <- 1 
dtm_title <- dtm.generate(corpus_title, n) 
dtm_text <- dtm.generate(corpus_text,n)

sparse <- 0.995 
df_title <- dtm.to.df(dtm_title, sparse, "title")
df_text <- dtm.to.df(dtm_text, sparse, "text")
data <- raw_label %>% 
        inner_join(df_text, by="DocID") %>% 
        inner_join(df_title, by="DocID") %>% 
        select(-DocID) 

train <- data[inTrain,]
test <- data[-inTrain,]

model <- naiveBayes(FakeorRealLabel~., data=train)
pred <- predict(model, test, type="class")
out <- class.out(pred, test$FakeorRealLabel)

out$ClassMatrix
out$Accuracy
