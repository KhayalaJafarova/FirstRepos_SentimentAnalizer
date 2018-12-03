import re
import nltk
import csv
#import nltk
#from sklearn.linear_model import NBClassifier

def processSentence(sentence):
    sentence = sentence.lower()
    return sentence

#Read the tweets one by one and process it
fp = open('C:/Users/Administrator/Desktop/Analizer/combine1.csv', 'r', encoding="utf8")
line = fp.readline()

while line:
    processedSentence = processSentence(line)
    #print(processedSentence)
    line = fp.readline()
#end loop
fp.close()

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getfeatureVector
def getFeatureVector(sentence):
    featureVector = []
    #split tweet into words
    words = sentence.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word starts with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        featureVector.append(w.lower())
    return featureVector
#end

#Read the tweets one by one and process it
fp = open('C:/Users/Administrator/Desktop/Analizer/combine1.csv', 'r', encoding="utf8")
line = fp.readline()


while line:
    processedSentence = processSentence(line)
    featureVector = getFeatureVector(processedSentence)
    #print(featureVector)
    line = fp.readline()
#end loop
fp.close()

#Read the tweets one by one and process it
inputSentences = csv.reader(open('C:/Users/Administrator/Desktop/Analizer/combine1.csv', 'r', encoding="utf8"), delimiter=',', quotechar='|')
featureList = []

sentences = []


for row in inputSentences:
    sentence = row[0]
    label = row[1]
    processedSentence = processSentence(sentence)
    featureVector = getFeatureVector(processedSentence)
    featureList.extend(featureVector)
    sentences.append((featureVector, label));

    #print(featureList)
#end

#start extract_features
def extract_features(sentence):
    sentence_words = set(sentence)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in sentence_words)
    return features
#end
#Read the tweets one by one and process it
inputSentences = csv.reader(open('C:/Users/Administrator/Desktop/Analizer/combine1.csv', 'r', encoding= "utf8"), delimiter=',', quotechar='|')

featureList = []

# Get tweet words
sentences = []
for row in inputSentences:
    label = row[1]
    sentence = row[0]
    processedSentence = processSentence(sentence)
    featureVector = getFeatureVector(processedSentence)
    featureList.extend(featureVector)
    sentences.append((featureVector, label));
#end loop

# Remove featureList duplicates
featureList = list(set(featureList))

# Extract feature vector for all tweets in one shote
training_set = nltk.classify.util.apply_features(extract_features, sentences)

# Train the classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

# Test the classifier
testSentence = 'çirkindi'
processedTestSentence = processSentence(testSentence)
print(NBClassifier.classify(extract_features(getFeatureVector(processedTestSentence))))