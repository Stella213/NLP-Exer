### Start with loading all necessary libraries
import nltk
nltk.download('stopwords')

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk import FreqDist
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer

def get_error_type(pred, label):
    # return the type of error: tp,fp,tn,fn
    if pred == 1 and label == 1:
        return 'tp'
    elif pred == 1 and label == 0:
        return 'fp'
    elif pred == 0 and label == 0:
        return 'tn'
    else:
        return 'fn'

    
### Text normalization
#Tokenization of text
tokenizer=ToktokTokenizer()


### 1. Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

### 2. Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)  

### 3. Removing stopwords

#Setting English stopwords
#stopword_list=nltk.corpus.stopwords.words('english') ###0.61796
stopword_list=["a","about","above","after","again","against","ain","all","am","an","and","any",
               "are","aren","aren't","as","at","be","because","been","before","being","below",
               "between","both","but","by","can","couldn","couldn't","d","did","didn","didn't",
               "do","does","doesn","doesn't","doing","don","don't","down","during","each","few",
               "for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven",
               "haven't","having","he","her","here","hers","herself","him","himself","his","how",
               "i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m",
               "ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn",
               "needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our",
               "ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's",
               "should","should've","shouldn","shouldn't","so","some","such","t","than","that",
               "that'll","the","their","theirs","them","themselves","then","there","these","they",
               "this","those","through","to","too","under","until","up","ve","very","was","wasn",
               "wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]


# removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [
            token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text



### 4. Removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    return text



### 5. Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text


### remove 1,2,3,4,5
def denoise(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)    
    text = remove_special_characters(text)
    text = simple_stemmer(text)
    return text



### classify
def classify(text, inqtabs_dict, swn_dict):
    x = ['enjoy','enjoyable','perfectly','perfect','implausible','wonderful','inspiration',
          'inspired','wonderfully','amusing','loved','best','historically','beautifully',
          'wondrous','well-developed','praised']
    y = ['terrible','boring','wasted']
    in_list = []
    swn_list = []
    
    with open('/Users/zhuoer/Downloads/nlp-hw1/lexicon/afinn.txt') as f:
        afinn_d = {}
        for line in f.readlines():
            if len(line) > 1:
                line = line.strip()
                k,v = line.split('\t')
                afinn_d[k] = v
    f.close()
    
    afinn_l = []
    for word in text.split(' '):
        try:
            afinn_l.append(int(afinn_d.get(word,0)))
        except:
            pass
    
    if word in x:
        return 1
    elif word in y:
        return 0
    elif np.sum(np.array(afinn_l)) > 3:
        return 1
    else:
        return 0

