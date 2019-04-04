from generate_data import generate_train_and_test_data
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import nltk
from sklearn.preprocessing import LabelBinarizer

#this function calculates the zscore for the test text
#it takes the text and counts the probabilities for common words
#and uses the frequentWordsCorpusMean and frequentWordsCorpusStdDev
def calc_z_score(text,frequentWordsCorpusMean,frequentWordsCorpusStdDev):
    word_counts=defaultdict(int)
    totalWords=0
    sentences=sent_tokenize(text.lower())
    for sentence in sentences:
        words=word_tokenize(sentence)
        fdist = nltk.FreqDist(words)
        for word in frequentWordsCorpusMean:
            if word in fdist:
                word_counts[word]+=fdist[word]
            else:
                word_counts[word]+=0
            totalWords+=fdist[word]
    zScores={}
    for word in word_counts:
        word_dist=(word_counts[word]+0.000001)/(totalWords+0.000001)
        zScores[word]=(word_dist-frequentWordsCorpusMean[word])/frequentWordsCorpusStdDev[word]
    return zScores

def find_email_match(text,frequentWordsCorpusMean,frequentWordsCorpusStdDev):
    scores={}
    min_score=1000000
    min_name=''
    zscores=calc_z_score(text,frequentWordsCorpusMean,frequentWordsCorpusStdDev)
    for author in zScoresByAuthor:
        score=0.0
        for word in zScoresByAuthor[author]:
            score+=abs(zscores[word]-zScoresByAuthor[author][word])
        score/=len(zScoresByAuthor[author])
        scores[author]=score
        if score<min_score:
            min_score=score
            min_name=author
    return min_name,min_score,scores

def softmax(x):
    e_x=np.exp(x-np.max(x))
    out=e_x/e_x.sum()
    return out

def cross_entropy_loss(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def get_probs(text,encoded_classes,frequentWordsCorpusMean,frequentWordsCorpusStdDev):
    zscores=calc_z_score(text,frequentWordsCorpusMean,frequentWordsCorpusStdDev)
    returnMatrix=[0.0 for author in zScoresByAuthor]
    for author in zScoresByAuthor:
        score=0.0
        for word in zScoresByAuthor[author]:
            score+=abs(zscores[word]-zScoresByAuthor[author][word])
        score/=len(zScoresByAuthor[author])
        for i,item in enumerate(encoded_classes):
            if item==author: returnMatrix[i]=score
    returnMatrix=[-x for x in returnMatrix]
    return softmax(returnMatrix),returnMatrix

filter_list=['kay.mann@enron.com','vince.kaminski@enron.com','jeff.dasovich@enron.com',
                 'chris.germany@enron.com','sara.shackleton@enron.com','tana.jones@enron.com',
                'eric.bass@enron.com','matthew.lenhart@enron.com','kate.symes@enron.com','sally.beck@enron.com']

min_words=[i for i in range(0,100,10)]
most_frequent = [i for i in range(10,100,10)]
for min_word in min_words:
    df, df_train, df_test = generate_train_and_test_data('data/emails.csv', filter_list, min_words=min_word)
    for n_most_frequent in most_frequent:
        # define a dictionary to hold the word count for each author
        author_subcorpus_count = {}
        for item in filter_list:
            author_subcorpus_count[item] = defaultdict(int)

        all_text = df_train['FormattedMessage'].tolist()
        from_list = df_train['From'].tolist()

        # word counts for the combined corpus
        word_counts = defaultdict(int)

        # go through the entire corpus, count words for the combined corpus and for each author
        for i, text in enumerate(all_text):
            sentences = sent_tokenize(text.lower())
            for sentence in sentences:
                words = word_tokenize(sentence)
                fdist = nltk.FreqDist(words)
                for word in fdist:
                    word_counts[word] += fdist[word]
                    author_subcorpus_count[from_list[i]][word] += fdist[word]

        # create a list of most frequent words
        # also check what total word count is (to validate data)
        freq_list = []
        i = 0
        totalWords = 0
        for w in word_counts:
            totalWords += word_counts[w]
        for w in sorted(word_counts, key=word_counts.get, reverse=True):
            if i < n_most_frequent: freq_list.append((w, word_counts[w]))
            i += 1

        # aggregate total words by author
        # ensure it adds up to total words by corpus
        totalWordsByAuthor = {}
        totalWords = 0
        for author in author_subcorpus_count:
            totalWordsByAuthor[author] = sum(author_subcorpus_count[author][x] for x in author_subcorpus_count[author])
            totalWords += totalWordsByAuthor[author]

        # we compute the mean for the corpus 2 ways
        # by corpus - so for example count "the" in the entire corpus/ total words in corpus
        # or compute the prob of "the" in each author's corpus and average it
        # the 2 results are not that different
        frequentWordsCorpusMean = {}
        frequentWordsCorpusStdDev = {}
        for word, count in freq_list:
            frequentWordsCorpusMean[word] = (count + 0.000001) / totalWords
            frequentWordsCorpusStdDev[word] = 0.0

        topWordsByAuthor = {}
        for item in author_subcorpus_count:
            topWordsByAuthor[item] = {}
            for word, count in freq_list:
                wc = author_subcorpus_count[item][word]
                wp = (wc + 0.000001) / totalWordsByAuthor[item]
                topWordsByAuthor[item][word] = wp

        frequentWordsMean = {}
        for word, count in freq_list:
            frequentWordsMean[word] = 0.0
            for author in topWordsByAuthor:
                frequentWordsMean[word] += topWordsByAuthor[author][word]
            frequentWordsMean[word] /= len(topWordsByAuthor)

        for word, count in freq_list:
            for author in topWordsByAuthor:
                diff = topWordsByAuthor[author][word] - frequentWordsCorpusMean[word]
                frequentWordsCorpusStdDev[word] += diff * diff
            frequentWordsCorpusStdDev[word] /= len(topWordsByAuthor)
            frequentWordsCorpusStdDev[word] = math.sqrt(frequentWordsCorpusStdDev[word])

        # calculate zscores
        # for each author, calculate the zscore for each of the common words
        zScoresByAuthor = {}
        for author in topWordsByAuthor:
            zScoresByAuthor[author] = {}
            for word in frequentWordsCorpusMean:
                zScoresByAuthor[author][word] = (topWordsByAuthor[author][word] - frequentWordsCorpusMean[word]) / (
                            frequentWordsCorpusStdDev[word] + 0.00001)

        enc = LabelBinarizer()
        enc.fit(filter_list)
        all_text = df_test['FormattedMessage'].tolist()
        from_list = df_test['From'].tolist()
        y_values = enc.transform(from_list)
        y_pred = []
        for text in all_text:
            prob, blah = get_probs(text, enc.classes_, frequentWordsCorpusMean, frequentWordsCorpusStdDev)
            y_pred.append(prob)

        y_pred = np.array(y_pred)
        print(min_word,n_most_frequent,cross_entropy_loss(y_values, y_pred))