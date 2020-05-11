import re
import sys
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import string

GLOVE_DIMENSION = 25
MAX_WORDS = 30 

CONTRACTIONS = { 
    "ain't": "am not",
    "aren't": "are not",
    "amn't": "am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "daren't": "dare not",
    "daresn't": "dare not",
    "dasn't": "dare not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "e'er": "ever",
    "everyone's": "everyone is",
    "finna": "going to",
    "gimme": "give me",
    "giv'n": "given",
    "gonna": "going to",
    "gon't": "go not",
    "gotta": "got to",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",    
    "he've": "he have",
    "howdy": "how do you do",
    "how're": "how are",
    "i'd": "I had",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "i'm'a": "I am about to",
    "i'm'o": "I am going to",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "ne'er": "never",
    "o'clock": "of the clock",
    "o'er": "over",
    "ol'": "old",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "somebody's": "somebody is",
    "someone's": "someone is",
    "something's": "something is",
    "so've": "so have",
    "so's": "so is",
    "so're": "so are", 
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "'tis": "it is",
    "'twas": "it was",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

ABBREVIATIONS = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

'''Function to preprocess the tweets'''
def preprocess_tweet(tweet):
    # MULTILINE - '^' matches the beggining of each line
    # DOTALL - '.' matches every character including newline
    FLAGS = re.MULTILINE | re.DOTALL
          
    # Replace links with token <url>
    tweet = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ", tweet, flags = FLAGS)
    
    # Remove hashtags
    tweet = re.sub(r"\#","", tweet, flags = FLAGS)
    
    # Remove mentions, starting with @
    tweet = re.sub(r'@\w+', '', tweet, flags = FLAGS)
            
    # Eyes of a smiley can be represented with: 8:=;
    # Nose of a smiley can be represented with: '`\-
    
    # Replace smiling face with <smile>. Mouth can be repredented with: )dD.
    tweet = re.sub(r"[8:=;]['`\-]?[)dD]+|[(dD]+['`\-]?[8:=;]", " <smile> ", tweet, flags = FLAGS)
    
    # Replace lol face with <lolface>. Mouth can be represented with: pP
    tweet = re.sub(r"[8:=;]['`\-]?[pP]+", " <lolface> ", tweet, flags = FLAGS)
    
    # Replace sad face with <sadface>. Mouth can be represented with: (
    tweet = re.sub(r"[8:=;]['`\-]?[(]+|[)]+['`\-]?[8:=;]", " <sadface> ", tweet, flags = FLAGS)
    
    # Replace neutral face with <neutralface>. Mouth can be represented with: \/|l
    tweet = re.sub(r"[8:=;]['`\-]?[\/|l]+", " <neutralface>", tweet, flags = FLAGS)
    
    # Split concatenated words wih /. Ex. Good/Bad -> Good Bad
    tweet = re.sub(r"/"," / ", tweet, flags = FLAGS)
    
    # Replace <3 with <heart>
    tweet = re.sub(r"<3"," <heart> ", tweet, flags = FLAGS)
    
    # Replace numbers with <number>.
    tweet = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", tweet, flags = FLAGS)
    
    # Replace repeated punctuation with <repeat>. Ex. !!! -> ! <repeat>
    tweet = re.sub(r"([!?.]){2,}", r"\1 <repeat> ", tweet, flags = FLAGS)
    
     # Replace elongated endings with <elong>. Ex. happyyy -> happy <elong>
    tweet = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ", tweet, flags = FLAGS)
       
    # Expand English contractions
    for word in tweet.split():
        if word.lower() in CONTRACTIONS:
            tweet = tweet.replace(word, CONTRACTIONS[word.lower()])
            
    # Expand abbreviations
    for word in tweet.split():
        if word.lower() in ABBREVIATIONS:
            tweet = tweet.replace(word, ABBREVIATIONS[word.lower()])
            
    # Remove apostrophes
    tweet = re.sub(r"'","", tweet, flags = FLAGS)

    # Add space between punctuation and word
    tweet = re.sub(r'(?<=[^ ])(?=[.,!?()])|(?<=[.,!?()])(?=[^ ])', r' ', tweet, flags = FLAGS)
    
    # Replace multiple empty spaces with one
    tweet = re.sub('\s+', ' ', tweet, flags = FLAGS)
       
    # Convert all tokens to lowercase
    tweet = tweet.lower()
    
    # Return result
    return tweet

def generate_embedding_matrix():
    df_train = pd.read_csv("../data/train.csv")
    df_test = pd.read_csv("../data/test.csv")
    
    train_tweets = []  
    test_tweets = []  
    labels = []
    
    # Preprocessing the train tweets
    for row in df_train.iterrows():
        tweet = preprocess_tweet(row[1]['text'])
        train_tweets.append(tweet)
        labels.append(row[1]['target'])
        
    y_train = labels
    
    # Preprocessing the test tweets
    for row in df_test.iterrows():
        tweet = preprocess_tweet(row[1]['text'])
        test_tweets.append(tweet)
        
    # Mapping every unique word to a integer (bulding the vocabulary)
    word_to_index = {}
    words_freq = {}
    m = 0
    
    for i, tweet in enumerate(train_tweets):
        words = tweet.split()
        
        for word in words[:MAX_WORDS]:
            if word not in word_to_index:
                word_to_index[word] = m
                m += 1
            if word not in words_freq:
                words_freq[word] = 1
            else:
                words_freq[word] += 1
            
    word_to_index["unk"] = m
    vocabulary_size = len(word_to_index)
            
    # Converting training tweets to integer sequences
    train_sequences = []

    for i, tweet in enumerate(train_tweets):
        words = tweet.split()

        tweet_seq = []
        for word in words[:MAX_WORDS]:
            if word not in word_to_index:
                tweet_seq.append(word_to_index["unk"])
            else:
                tweet_seq.append(word_to_index[word])

        train_sequences.append(tweet_seq)
    
    # Padding the sequences to match the `MAX_WORDS`
    X_train = pad_sequences(train_sequences, maxlen=MAX_WORDS, padding="post", value=vocabulary_size)
    
    # Converting test tweets to integer sequences
    test_sequences = []

    for i, tweet in enumerate(test_tweets):
        words = tweet.split()

        tweet_seq = []
        for word in words[:MAX_WORDS]:
            if word not in word_to_index:
                tweet_seq.append(word_to_index["unk"])
            else:
                tweet_seq.append(word_to_index[word])

        test_sequences.append(tweet_seq)
    
    # Padding the sequences to match the `MAX_WORDS`
    X_test = pad_sequences(test_sequences, maxlen=MAX_WORDS, padding="post", value=vocabulary_size)
    
   # Reading glove embeddings
    glove_embeddings_file = open('../data/glove.twitter.27B.25d.txt', 'r', encoding='UTF-8')

    glove_embeddings = dict()
    for line in glove_embeddings_file:
        parts = line.split()
        key = parts[0]
        embedding = [float(t) for t in parts[1:]]
        glove_embeddings[key] = np.array(embedding)

    # Generating the embedding matrix for our vocabulary (this is needed for the Embedding layer in keras models)
    unknown = []
    hits = 0
    embedding_matrix = np.zeros((vocabulary_size + 1, GLOVE_DIMENSION))
    for word, idx in word_to_index.items():
        if word in glove_embeddings:
            emb = glove_embeddings[word]
            embedding_matrix[idx] = emb
            hits += 1
    else:
        unknown.append(word)
        emb = glove_embeddings["unk"]
        embedding_matrix[idx] = emb
    
    embedding_matrix[vocabulary_size] = [0]*GLOVE_DIMENSION
        
    return (X_train, y_train, X_test, embedding_matrix)


'''Function to remove English stopwords from a Pandas Series.'''
def remove_stopwords(input_text):
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words) 



'''Function to preprocess the tweets'''
def preprocess_tweet_use(tweet):
    # MULTILINE - '^' matches the beggining of each line
    # DOTALL - '.' matches every character including newline
    FLAGS = re.MULTILINE | re.DOTALL
          
    # Replace links
    tweet = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " ", tweet, flags = FLAGS)
    
    # Remove hashtags
    tweet = re.sub(r"\#","", tweet, flags = FLAGS)
    
    # Remove mentions, starting with @
    tweet = re.sub(r'@\w+', '', tweet, flags = FLAGS)
            
    # Eyes of a smiley can be represented with: 8:=;
    # Nose of a smiley can be represented with: '`\-
    
    # Replace smiling face with <smile>. Mouth can be repredented with: )dD.
    tweet = re.sub(r"[8:=;]['`\-]?[)dD]+|[(dD]+['`\-]?[8:=;]", " smile ", tweet, flags = FLAGS)
    
    # Replace lol face with <lolface>. Mouth can be represented with: pP
    tweet = re.sub(r"[8:=;]['`\-]?[pP]+", " lol ", tweet, flags = FLAGS)
    
    # Replace sad face with <sadface>. Mouth can be represented with: (
    tweet = re.sub(r"[8:=;]['`\-]?[(]+|[)]+['`\-]?[8:=;]", " sad ", tweet, flags = FLAGS)
    
    # Replace neutral face with <neutralface>. Mouth can be represented with: \/|l
    tweet = re.sub(r"[8:=;]['`\-]?[\/|l]+", " neutral", tweet, flags = FLAGS)
    
    # Split concatenated words wih /. Ex. Good/Bad -> Good Bad
    tweet = re.sub(r"/"," / ", tweet, flags = FLAGS)
    
    # Replace <3 with <heart>
    tweet = re.sub(r"<3"," heart ", tweet, flags = FLAGS)
    
    # Replace elongated endings with <elong>. Ex. happyyy -> happy <elong>
    tweet = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ", tweet, flags = FLAGS)
    
    # Remove <elong>
    tweet = re.sub(r"<elong>", r" ", tweet, flags = FLAGS)
    
    # Expand English contractions
    for word in tweet.split():
        if word.lower() in CONTRACTIONS:
            tweet = tweet.replace(word, CONTRACTIONS[word.lower()])
            
    # Expand abbreviations
    for word in tweet.split():
        if word.lower() in ABBREVIATIONS:
            tweet = tweet.replace(word, ABBREVIATIONS[word.lower()])

    # Remove punctuation
    tweet = tweet.strip(string.punctuation)
    
    # Replace multiple empty spaces with one
    tweet = re.sub('\s+', " ", tweet, flags = FLAGS)
       
    # Convert all tokens to lowercase
    tweet = tweet.lower()
    
    # Return result
    return tweet