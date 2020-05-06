import re

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


'''Function to preprocess the tweets'''
def preprocess_tweet(tweet):
    # MULTILINE - '^' matches the beggining of each line
    # DOTALL - '.' matches every character including newline
    FLAGS = re.MULTILINE | re.DOTALL
    
    # Replace links with token <url>
    tweet = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ", tweet, flags = FLAGS)
    
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
    
    # Replace multiple empty spaces with one
    tweet = re.sub('\s+', ' ', tweet, flags = FLAGS)
    
    # Expand English contractions
    for word in tweet.split():
        if word.lower() in CONTRACTIONS:
            tweet = tweet.replace(word, CONTRACTIONS[word.lower()])
    
    # Remove apostrophes
    tweet = re.sub(r"'","", tweet, flags = FLAGS)
    
    # Remove mentions, starting with @
    tweet = re.sub(r'@\w+', '', tweet, flags = FLAGS)
    
     # convert all tokens to lowercase
    tweet = tweet.lower()
    
    # Return result
    return tweet