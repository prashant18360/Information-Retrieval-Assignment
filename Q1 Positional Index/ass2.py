import glob
import os.path
from os import path
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
from os.path import split
from os.path import basename
import ntpath
import pickle
import sys
#this is only for serializing the class object by pickle dump
sys.setrecursionlimit(10000)


class mainfun:
    def __init__(self):
        self.docID = 0
        self.p_index = {}
        self.docname = []
    def preprocessing(self,content):
        content=content.lower()
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        content_tokens = tokenizer.tokenize(content)
        r_1=[]
        stop_words = set(stopwords.words('english'))
        for w in content_tokens:
            if w not in stop_words:
                r_1.append(w)

        r_2 =[t for t in r_1 if t not in string.punctuation]

        r_3=[]
        for d in r_2:
            if d!=" ":
                r_3.append(d)
        stemmer = PorterStemmer()
        result=[]
        for x in r_3:
            x=stemmer.stem(x)
            result.append(x)

        return result

    # Extracting the Document name from path of a document
    def fnamebypath(self, fpath):
        first, last = ntpath.split(fpath)
        return last or ntpath.basename(first)


if __name__ == '__main__':

    # set the directory's path of dataset
    directorypath = 'D:\IIITD\SEMESTER 6\Information Retrieval\Assignment\A2\stories\**\*'

    # creating the class object
    mainobj = mainfun()
    x = 1

    # Here golb is used for the finding the all file path in a directory and also the inner directory file path
    for filepath in glob.glob(directorypath, recursive=True):

        if (path.isfile(filepath) and not filepath.endswith('.html')):
            # Calling the function which extract filename from path of a file and then store in a docname list

            # Calling the function which extract filename from path of a file and then store in a docname list
            filename = mainobj.fnamebypath(filepath)
            mainobj.docname.append(filename)

            # Reading of file
            with open(filepath, 'rb') as file:
                filecontent = file.read().decode(errors='replace')

            preprocessresult = {}
            # Calling the function of preprocessing of file content
            preprocessresult = mainobj.preprocessing(filecontent)
            # making positional index
            i=0
            for term in (preprocessresult):
                i+=1
                if term not in mainobj.p_index:
                    mainobj.p_index[term]=[]
                    mainobj.p_index[term].append(1)
                    mainobj.p_index[term].append({})
                    mainobj.p_index[term][1][mainobj.docID]=[i]
                else:
                    x=mainobj.p_index[term][0]
                    mainobj.p_index[term][0]=x+1
                    if(mainobj.docID not in mainobj.p_index[term][1]):
                        mainobj.p_index[term][1][mainobj.docID] = [i]
                    else:
                        mainobj.p_index[term][1][mainobj.docID].append(i)

            mainobj.docID = mainobj.docID +1
    with open('store.dat', 'wb') as fp:
        pickle.dump(mainobj, fp)
