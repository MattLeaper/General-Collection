#!/usr/bin/env python
import re, random, math, collections, itertools
import pandas as pd
import numpy as np

PRINT_ERRORS=0

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())
 
    with open('positive-words.txt', 'r', encoding="ISO-8859-1") as posDictionary:
        posWordList = posDictionary.read().splitlines()

    with open('negative-words.txt', 'r', encoding="ISO-8859-1") as negDictionary:
        negWordList = negDictionary.read().splitlines()

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    # Create Training and Test Datsets:
    # We want to test on sentences we haven't trained on, 
    # to see how well the model generalses to previously unseen sentences

    # create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    # create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

# calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    # iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: # calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                # keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1 # keeps count of total words in negative class
                
                # keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        # do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

# implement naive bayes algorithm
# INPUTS:
#   sentencesTest is a dictonary with sentences associated with sentiment 
#   dataName is a string (used only for printing output)
#   pWordPos is dictionary storing p(word|positive) for each word
#      i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#   pWordNeg is dictionary storing p(word|negative) for each word
#   pWord is dictionary storing p(word)
#   pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):

    print("Naive Bayes classification")
    pNeg=1-pPos

    # These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    # for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: # calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)

    #calculate accuracy
    accuracy = correct / total

    #Precision and Recall for positive 
    precision_pos = correctpos / totalpospred if totalpospred > 0 else 0
    recall_pos = correctpos / totalpos if totalpos > 0 else 0
    f1_pos = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

    #Precision and Recall for negative 
    precision_neg = correctneg / totalnegpred if totalnegpred > 0 else 0
    recall_neg = correctneg / totalneg if totalneg > 0 else 0
    f1_neg = (2 * precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0


    #printing out results for step 2
    print(f"Results for {dataName}:")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision (pos): {precision_pos:.4f}, Recall (pos): {recall_pos:.4f}, F1 (pos): {f1_pos:.4f}")
    print(f"Precision (neg): {precision_neg:.4f}, Recall (neg): {recall_neg:.4f}, F1 (neg): {f1_neg:.4f}")


# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):

    print("Dictionary-based classification")
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1


    #calculates accuracy
    accuracy = correct / total

    #Precision and Recall for positive 
    precision_pos = correctpos / totalpospred if totalpospred > 0 else 0
    recall_pos = correctpos / totalpos if totalpos > 0 else 0
    f1_pos = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

    #Precision and Recall for negative 
    precision_neg = correctneg / totalnegpred if totalnegpred > 0 else 0
    recall_neg = correctneg / totalneg if totalneg > 0 else 0
    f1_neg = (2 * precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0

    #printing out results for accuracy, precision, recall and f1 for step 5
    print(f"Results for {dataName}:")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision (pos): {precision_pos:.4f}, Recall (pos): {recall_pos:.4f}, F1 (pos): {f1_pos:.4f}")
    print(f"Precision (neg): {precision_neg:.4f}, Recall (neg): {recall_neg:.4f}, F1 (neg): {f1_neg:.4f}")

def rule_based_system(sentences_test, data_name, sentiment_dictionary, threshold=0.05):
    negation_words = []
    with open("negation_words.txt", 'r') as f:
        negation_words = [line.strip() for line in f]
    strengthen_words = pd.read_csv("strengthen_words.tsv", sep='\t', index_col=0)['score'].to_dict()
    weaken_words = pd.read_csv("weaken_words.tsv", sep='\t', index_col=0)['score'].to_dict()
    total = 0
    correct = 0
    totalpos = 0
    totalneg = 0
    totalpospred = 0
    totalnegpred = 0
    correctpos = 0
    correctneg = 0
    def calculate_valence(sentence):
        # Tokenize words while preserving the structure of sentences
        words = re.findall(r"[\w']+", sentence.lower())  # Lowercased to ensure consistent matching
        valence = 0
        negated = False
        context_multiplier = 1.2 # Adjust sentiment for strengthen/weaken words

        for i, word in enumerate(words):
            # Sentiment contribution from the dictionary
            if word in sentiment_dictionary:
                current_valence = sentiment_dictionary[word]
                current_valence *= context_multiplier  # Apply context scaling
                valence += -current_valence if negated else current_valence

            # Handle negations (apply to the next sentiment word)
            if word in negation_words:
                negated = True
            elif negated and word not in sentiment_dictionary:  # Reset negation after neutral word
                negated = False

            # Strengthen words amplify sentiment
            if word in strengthen_words:
                context_multiplier *= (1 + strengthen_words[word])

            # Weaken words reduce sentiment intensity
            elif word in weaken_words:
                context_multiplier *= (1 - weaken_words[word])

        if sentence.isupper():
            valence *= 1.5
        return valence
    for sentence, true_sentiment in sentences_test.items():
        total += 1
        score = calculate_valence(sentence)
        if score >= threshold:
            predicted_sentiment = 'positive'
        elif score <= -threshold:
            predicted_sentiment = 'negative'
        else:
            predicted_sentiment = 'neutral'

        if true_sentiment == "positive":
            totalpos += 1
            if predicted_sentiment == "positive":
                correct += 1
                correctpos += 1
                totalpospred += 1
            else:
                totalnegpred += 1
        elif true_sentiment == "negative":
            totalneg += 1
            if predicted_sentiment == "negative":
                correct += 1
                correctneg += 1
                totalnegpred += 1
            else:
                totalpospred += 1

    # Calculate accuracy
    accuracy = correct / total 
    # Calculate Precision, Recall, F1 for positive sentiment
    precision_pos = correctpos / totalpospred 
    recall_pos = correctpos / totalpos
    f1_pos = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos)
    # Calculate Precision, Recall, F1 for negative sentiment
    precision_neg = correctneg / totalnegpred 
    recall_neg = correctneg / totalneg 
    f1_neg = (2 * precision_neg * recall_neg) / (precision_neg + recall_neg) 
    # Print out the results
    print(f"Results for {data_name}:")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision (pos): {precision_pos:.4f}, Recall (pos): {recall_pos:.4f}, F1 (pos): {f1_pos:.4f}")
    print(f"Precision (neg): {precision_neg:.4f}, Recall (neg): {recall_neg:.4f}, F1 (neg): {f1_neg:.4f}")


def mostUseful(pWordPos, pWordNeg, pWord, n):
    # Create a dictionary to store the prediction power of each word
    predictPower = {}

    # Calculate prediction power for each word based on its probability in the positive and negative classes
    for word in pWord:
        if pWordNeg[word] < 0.0000001:
            predictPower[word] = 1000000000  # To avoid division by zero or a very small number
        else:
            predictPower[word] = pWordPos[word] / (pWordPos[word] + pWordNeg[word])

    # Sort the words by their prediction power (positive words first)
    sortedPower = sorted(predictPower, key=predictPower.get, reverse=True)  # Sort in descending order

    # Get the top n and bottom n words based on prediction power
    head, tail = sortedPower[:n], sortedPower[-n:]

    # Print the most useful words for positive and negative classes
    print("NEGATIVE:")
    print(tail)  # Bottom words (most negative)
    print("\nPOSITIVE:")
    print(head)  # Top words (most positive)

    with open('positive-words.txt', 'r', encoding="ISO-8859-1") as posDictionary:
        posWordList = posDictionary.read().splitlines()

    with open('negative-words.txt', 'r', encoding="ISO-8859-1") as negDictionary:
        negWordList = negDictionary.read().splitlines()
    
    
    posmatch = sum(1 for word in head if word in posWordList)
    print(f"Positive matches: {posmatch}")

    negmatch = sum(1 for word in tail if word in negWordList)
    print(f"Negative matches: {negmatch}")


#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

# build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

# run naive bayes classifier on datasets
#testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
#testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
#testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)

rule_based_system(sentencesNokia, "Nokia   (All Data, Improved Rule-Based)\t",  sentimentDictionary, 1)
rule_based_system(sentencesTrain,  "Films (Train Data, Improved Rule-Based)\t", sentimentDictionary, 1)
rule_based_system(sentencesTest,  "Films  (Test Data, Improved Rule-Based)\t",  sentimentDictionary, 1)


# run sentiment dictionary based classifier on datasets
#testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
#testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1)
#testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1)


#print most useful words
#mostUseful(pWordPos, pWordNeg, pWord, 100)
