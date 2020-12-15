import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

punctuations = string.punctuation
stopwords_list = stopwords.words('english')
lemm = WordNetLemmatizer()

def clean_text(text):
    cleaned_text = ''
    cleaned_text = text.lower()
    cleaned_text = ''.join(c for c in cleaned_text if c not in punctuations)
    
    words = cleaned_text.split()
    words = [word for word in words if word not in stopwords_list]
    
    #words = [word for word in words if not word.isdigit()]
    
    words = [lemm.lemmatize(word, 'v') for word in words]
    words = [lemm.lemmatize(word, 'n') for word in words]
    
    cleaned_text = ' '.join(words)
    
    return cleaned_text

def pos_check(txt, family):
    pos_dic = {"noun": ["NNP", "NN", "NNS", "NNPS"], "verb":["VBZ", "VBP", "VB", "VBD", "VBN"]}
    tags = nltk.pos_tag(nltk.word_tokenize(txt))
    count = 0
    for tag in tags:
        tag = tag[1]
        if tag in pos_dic[family]:
            count += 1
    return count

def create_dataframe(text):
    
    #cleaned_text = clean_text(text)
    
    d = {'text': [text]}
    df = pd.DataFrame(data=d)
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    df['word_count_wit_space'] = df['text'].apply(lambda x: len(x.split()))
    df['word_count_cleaned_wit_space'] = df['cleaned_text'].apply(lambda x: len(x.split()))

    #word count with no space
    df['word_count_wit_nospace'] = df['text'].apply(lambda x: len(x.replace(' ', '')))
    df['word_count_cleaned_wit_nospace'] = df['cleaned_text'].apply(lambda x: len(x.replace(' ', '')))

    #no of digit
    df["no_of_digits"] = df['text'].apply(lambda x: len([word for word in x.split() if word.isdigit()]))
    df["noun_count"] = df["text"].apply(lambda x: pos_check(x, "noun"))
    df["verb_count"] = df["text"].apply(lambda x: pos_check(x, "verb"))
    
    #df2 = df.copy()
    #df.drop(['text'], inplace=True, axis=1)
    
    return df




