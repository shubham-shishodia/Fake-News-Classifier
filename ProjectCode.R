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
