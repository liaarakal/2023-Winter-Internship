import pandas as pd
import re
import os

import gensim
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import remove_stopword_tokens
from gensim.corpora import Dictionary
from gensim import corpora

import nltk
from nltk.stem import WordNetLemmatizer
from gensim.parsing.porter import PorterStemmer
from gensim.models import LdaModel
import datetime

import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import seaborn as sn
nltk.download('omw-1.4')
nltk.download('wordnet') 
nltk.download('averaged_perceptron_tagger')



# Preprocessing params
# Can mess around with these for data analysis
l_lim_docs = 0
h_lim_docs = 40911 # max = 40911
regex_filtering = True
lemmatization = True # False will switch to stemming
occurance_min_count = 20
occurance_max_percent = .5
words_per_topic = 10 # Max 20
top_docs_per_topic =  10
show_docs = True

# LDA params
# Ideal number of topics fell between 15 - 25
num_topics = 15

# Only need to change if you delete previous save file
chunksize = 2000
passes = 20
iterations = 1000
eval_every = None

x = 0
j = 0

# Save files
model_save = str(l_lim_docs)+"_"+str(h_lim_docs)+"LDAfull"+"_"
if lemmatization:
    model_save = model_save+"L"+str(num_topics)+"_"+str(occurance_min_count)+"_"+str(int(occurance_max_percent*100))
else:
    model_save = model_save+"S"+str(num_topics)+"_"+str(occurance_min_count)+"_"+str(int(occurance_max_percent*100))

csv_save = str(l_lim_docs)+"_"+str(h_lim_docs)
csv_saveR = str(l_lim_docs)+"_"+str(h_lim_docs)+"R"
csv_saveRS = str(l_lim_docs)+"_"+str(h_lim_docs)+"RS"
csv_saveRSO = str(l_lim_docs)+"_"+str(h_lim_docs)+"RSO"
csv_saveRSOL = str(l_lim_docs)+"_"+str(h_lim_docs)+"RSOL"
csv_saveRSOS = str(l_lim_docs)+"_"+str(h_lim_docs)+"RSOS"
csv_saveRSOLT = str(l_lim_docs)+"_"+str(h_lim_docs)+"RSOLT"+str(num_topics)+"_"+str(occurance_min_count)+"_"+str(int(occurance_max_percent*100))
csv_saveRSOLTS = str(l_lim_docs)+"_"+str(h_lim_docs)+"RSOLTS"+str(num_topics)+"_"+str(occurance_min_count)+"_"+str(int(occurance_max_percent*100))
if lemmatization:
    doc_pickle = str(l_lim_docs)+"_"+str(h_lim_docs)+"_"+str(occurance_min_count)+"_"+str(int(occurance_max_percent*100))+'docL.pkl'
    corp_dict_pickle = str(l_lim_docs)+"_"+str(h_lim_docs)+"_"+str(occurance_min_count)+"_"+str(int(occurance_max_percent*100))+'corp_dictL.pkl'
    corpus_pickle = str(l_lim_docs)+"_"+str(h_lim_docs)+"_"+str(occurance_min_count)+"_"+str(int(occurance_max_percent*100))+'corpL.pkl'
else:
    doc_pickle = str(l_lim_docs)+"_"+str(h_lim_docs)+"_"+str(occurance_min_count)+"_"+str(int(occurance_max_percent*100))+'docS.pkl'
    corp_dict_pickle = str(l_lim_docs)+"_"+str(h_lim_docs)+"_"+str(occurance_min_count)+"_"+str(int(occurance_max_percent*100))+'corp_dictS.pkl'
    corpus_pickle = str(l_lim_docs)+"_"+str(h_lim_docs)+"_"+str(occurance_min_count)+"_"+str(int(occurance_max_percent*100))+'corpS.pkl'

re_currency = '(?i)\$(\w+)'
currency_pat = re.compile(re_currency)
re_energy = '(?i)^(m(w|\w+watt)|(t(w|\w+tt))|g(w|(\w+tt)))|watts?$'
energy_pat = re.compile(re_energy)
re_pv = '(?i)^pv$'

def regex_proc(doc):
    return re.sub(currency_pat,'currency',doc)

def regex_token_filter(doc):
    token = re.sub(re_pv,'photovoltaic',doc)
    return re.sub(re_energy,'energy',token)

def wn_lem(tokens):
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(word) for word in tokens]

def otherproc_remove(tokens):
    my_stop_words = set(["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz","'","\"","”","_","year","years","-","facebook","google","twitter","youtube","image","subscribe","subscriber","cleantechnica","–","—","it’s","lot"])
    if regex_filtering:
        return [regex_token_filter(word) for word in tokens if word not in my_stop_words]
    else:
        return [word for word in tokens if word not in my_stop_words]
    
def gm_stem(tokens):
    p_stem = PorterStemmer()
    new = []
    return [p_stem.stem(word) for word in tokens]
 
def openPKL():
    #if os.path.exists(csv_saveRSOLTS+".pkl"):
        #df = pd.read_pickle(csv_saveRSOLTS+".pkl")
        #stage = 7
    if os.path.exists(csv_saveRSOLT+".pkl"):
        df = pd.read_pickle(csv_saveRSOLT+".pkl")
        stage = 6
    elif os.path.exists(csv_saveRSOL+".pkl"):
        df = pd.read_pickle(csv_saveRSOL+".pkl")
        stage = 5
    
    elif os.path.exists(csv_saveRSOS+".pkl"):
        df = pd.read_pickle(csv_saveRSOS+".pkl")
        stage = 4
    
    elif os.path.exists(csv_saveRSO+".pkl"):
        df = pd.read_pickle(csv_saveRSO+".pkl")
        stage = 3
    
    elif os.path.exists(csv_saveRS+".pkl"):
        df = pd.read_pickle(csv_saveRS+".pkl")
        stage = 2
    
    elif os.path.exists(csv_saveR+".pkl"):
        df = pd.read_pickle(csv_saveR+".pkl")
        stage = 1
        
    else:
        df = pd.read_csv('solar_split_1.csv')
        df['textL'] = ""
        df['titleL'] = ""
        stage = 0
        
    return (stage,df)
 
def preprocessing_regex(df):  
    if regex_filtering and not os.path.exists(csv_saveR+".csv"):
        df.loc[l_lim_docs:h_lim_docs,'title'] = df.loc[l_lim_docs:h_lim_docs,'title'].apply(regex_proc)
        df.loc[l_lim_docs:h_lim_docs,'text'] = df.loc[l_lim_docs:h_lim_docs,'text'].apply(regex_proc)
        df.to_pickle(csv_saveR+".pkl")
    return df
def preprocessing_main(df):    
    if not os.path.exists(csv_saveRS+".csv"):
        df.loc[l_lim_docs:h_lim_docs,'titleL'] = df.loc[l_lim_docs:h_lim_docs,'title'].apply(preprocess_string,args=[[lambda x: x.lower(),strip_tags,strip_punctuation,strip_numeric,strip_multiple_whitespaces]])
        df.loc[l_lim_docs:h_lim_docs,'textL'] = df.loc[l_lim_docs:h_lim_docs,'text'].apply(preprocess_string,args=[[lambda x: x.lower(),strip_tags,strip_punctuation,strip_numeric,strip_multiple_whitespaces]])
        df.to_pickle(csv_saveRS+".pkl")
    return df

def preprocessing_other(df):
    if not os.path.exists(csv_saveRSO+".csv"):
        df.loc[l_lim_docs:h_lim_docs,'titleL'] = df.loc[l_lim_docs:h_lim_docs,'titleL'].apply(otherproc_remove)
        df.loc[l_lim_docs:h_lim_docs,'textL'] = df.loc[l_lim_docs:h_lim_docs,'textL'].apply(otherproc_remove)
        df.to_pickle(csv_saveRSO+".pkl")
    return df
 
def lem(df):
    if lemmatization and not os.path.exists(csv_saveRSOL+".csv"):
        df.loc[l_lim_docs:h_lim_docs,'textL'] = df.loc[l_lim_docs:h_lim_docs,'textL'].apply(wn_lem)
        df.loc[l_lim_docs:h_lim_docs,'titleL'] = df.loc[l_lim_docs:h_lim_docs,'titleL'].apply(wn_lem)
        df.to_pickle(csv_saveRSOL+".pkl")
    return df

def stem(df):
    if not lemmatization and not os.path.exists(csv_saveRSOS+".csv"):
        df.loc[l_lim_docs:h_lim_docs,'textL'] = df.loc[l_lim_docs:h_lim_docs,'textL'].apply(gm_stem)
        df.loc[l_lim_docs:h_lim_docs,'titleL'] = df.loc[l_lim_docs:h_lim_docs,'titleL'].apply(gm_stem)
        df.to_pickle(csv_saveRSOS+".pkl")
    return df

# Creates the model
def get_lda(corpus,corp_dict):
    model = ""
    if(os.path.exists(model_save)):
        model = LdaModel.load(model_save)
    else:
        temp = corp_dict[0]
        id2word = corp_dict.id2token
       
        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every)
        LdaModel.save(model,model_save)
    return model
 
def sort_by_date(df):
    df['published'] = df['published'].apply(pd.to_datetime)
    df.sort_values(by=['published'],ascending=False)
    df.to_pickle(csv_saveRSOLTS+".pkl")
    return df

# Generates a wordcloud visualization
# Need to manually add topic names under topics var
def wordcloud_gen(model):
    topics = ["Charging Ads","Community","Manufacturing","Investment","Astronomy","Systems design","Business","Energy Sources","Elon Musk","Markets","Development", "Meteorology", "Composition","Public sector","Infrastructure"]
    for t in range(model.num_topics):
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(model.show_topic(t, 200))))
        plt.axis("off")
        plt.title(topics[t])
        plt.show()

def one_doc_topics(doc,cd,model):
    bow = [cd.doc2bow(word) for word in [doc]]
    topics = model.get_document_topics(bow)[0]
    if(len(topics)<3):
        return topics
    else:
        highest = 0
        highestI = 0
        second = 0
        secondI = 0
        for i,topic in enumerate(topics):
            if topic[1]>highest:
                second = highest
                secondI = highestI
                highest = topic[1]
                highestI = i
            elif topic[1]>second:
                second = topic[1]
                secondI = i
        total = highest+second
        return [(highestI,(highest/total)),(secondI,(second/total))]

def all_doc_topics(df,cd,model):
    df['topics'] = ""
    df['topics'] = df['textL'].apply(one_doc_topics,args=[cd,model])
    return df

#row[0] = title, row[1] = text, row[2] = publish date, (!row[3] = id (document id)!),
#row[4] = 'titleL', row[5] = 'textL', (!row[6] = topics. [(topic,percentage),(topic,perctage)]!)
# Tldr just add tuples to dpt
def docs_for_topics(df,dpt):
    for i in df.itertuples():
        dpt[i[7][(0)][0]].append(([i[0], i[7][0][1]]))
        if(len(i[7])==2):
            dpt[i[7][(1)][0]].append(([i[0], i[7][1][1]]))
            
    for l in dpt:
        list.sort(l,key=lambda x: x[1],reverse=True)
    return dpt
  
def occurences(df,dpt):
    opt = [[0,0,0,0,0] for i in range(num_topics)]
    for i,topics in enumerate(dpt):
        for doc in topics:
            ar = {2015:0,2016:1,2017:2,2018:3,2020:4}
            yearVal=datetime.datetime.strptime(df.loc[doc[0],'published'],'%Y-%m-%dT%H:%M:%S.%f%z').year
            numDocumentsInYear=0;
            if(yearVal==2015):
                numDocumentsInYear=50.0/3812
            elif(yearVal==2016):
                numDocumentsInYear=50.0/16871
            elif(yearVal==2017):
                numDocumentsInYear=50.0/6251
            elif(yearVal==2018):
                numDocumentsInYear=50.0/2980
            elif(yearVal==2020):
                numDocumentsInYear=50.0/10997
            opt[i][ar[yearVal]]+=numDocumentsInYear
    return opt

# Creates a trending heat map over time
def plot_trending(opt):
    plt.title('Percent of documents about a topic each year',fontsize = 20)
    xticklabels = ["2015\n\n3812\nDocs","2016\n\n16871\nDocs","2017\n\n6251\nDocs","2018\n\n2980\nDocs","2020\n\n10997\nDocs",]
    yticklabels = ["Charging Ads","Community","Manufacturing","Investment","Astronomy","Systems design","Business","Energy Sources","Elon Musk","Markets","Development", "Meteorology", "Composition","Public sector","Infrastructure"]
    #yticklabels = ["Topic "+str(i+1) for i in range(num_topics)]
    hm = sn.heatmap(data=opt, xticklabels=xticklabels, yticklabels=yticklabels,annot=True,cmap='gnuplot2',)
    plt.xlabel('Years', fontsize = 15)
    plt.ylabel('Topics', fontsize = 15) 
    plt.show()
    
def main():
    (fstage,df) = openPKL()
    if(fstage<1):
        df = preprocessing_regex(df)
    if(fstage<2):
        df = preprocessing_main(df)
    if(fstage<3):
        df = preprocessing_other(df)
    if(fstage<4):
        if lemmatization:
            df = lem(df)
        else:
            df = stem(df)
    
    if not os.path.exists(doc_pickle):
        docs = df.loc[l_lim_docs:h_lim_docs,'textL']
        with open(doc_pickle, str('wb')) as file:
            pickle.dump(docs,file)
    else:
        with open(doc_pickle, str('rb')) as file:
            docs = pickle.load(file)
        
    if not os.path.exists(corp_dict_pickle):
        corp_dict = Dictionary(docs)
        corp_dict.filter_extremes(no_below=occurance_min_count, no_above=occurance_max_percent)
        with open(corp_dict_pickle, str('wb')) as file:
            pickle.dump(corp_dict, file)
    else:
        with open(corp_dict_pickle, str('rb')) as file:
            corp_dict = pickle.load(file)
    
    if not os.path.exists(corpus_pickle):
        corpus = [corp_dict.doc2bow(doc) for doc in docs]
        with open(corpus_pickle, str('wb')) as file:
            pickle.dump(corpus,file)
    else:
        with open(corpus_pickle, str('rb')) as file:
            corpus = pickle.load(file)
    
    model = get_lda(corpus,corp_dict)
    if fstage<6:
        df = all_doc_topics(df,corp_dict,model)
        df.to_pickle(csv_saveRSOLT+".pkl")
           
    wordcloud_gen(model) # Prints wordcloud visualization
    
    
    # This should be a list of list tuples (doc num, percent relevence) ex [[(1,.6),(2,.88)],[(6,.32)]....]
    documents_per_topic = [[] for i in range(num_topics)]
    documents_per_topic = docs_for_topics(df,documents_per_topic)
    for j,x in enumerate(documents_per_topic):
        print("Topic "+str(j))
        for i in range(len(x)):
            if show_docs:
                print(str(x[i][1])+"\n"+str(df.loc[x[i][0],'title'])+"\n"+str(df.loc[x[i][0],'text'])+"\n")
            else:
                print(str(x[i])+"\n")
            if(i==top_docs_per_topic):
                break
        print()
        
    trending = occurences(df, documents_per_topic)
    plot_trending(trending) # Prints the trending graph for given dataset

if __name__ == "__main__":
    main()
