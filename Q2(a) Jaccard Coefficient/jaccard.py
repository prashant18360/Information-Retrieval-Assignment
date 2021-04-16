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
        """This is the function  to covert text in lower case for removing the punctuation.
        word tokenization
        Also remove the  stop words sace tokens from  given files  . 
        Stemming over the tokens and converting     tokens into their base words.  """
    def preprocessing(self,content):
        # converting into lower case
        content=content.lower()
        #word tokenization.
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        content_tokens = tokenizer.tokenize(content)
        r_1=[]
        # collecting the stop words.
        stop_words = set(stopwords.words('english'))

        # Removing the stop words.
        for w in content_tokens:
            if w not in stop_words:
                r_1.append(w)
        # Removing of punctuation
        r_2 =[t for t in r_1 if t not in string.punctuation]

        r_3=[]
        #Word stemmeing.
        for d in r_2:
            if d!=" ":
                r_3.append(d)
        stemmer = PorterStemmer()
        result=[]
        for x in r_3:
            x=stemmer.stem(x)
            result.append(x)
        #returning the list containing the preprocessing result.
        return result
    def filename(self,  path):
        first, last=ntpath.split(path)
        return last or ntpath.basename(first)
if __name__ == '__main__':

    # set the directory's path of dataset
    directorypath = 'D:\IIITD\SEMESTER 6\Information Retrieval\Assignment\A2\stories\**\*'
    

    # creating the class object
    mainobj = mainfun()
    x = 1
    preprocessresult =[]

    # Here golb is used for the finding the all file path in a directory and also the inner directory file path
    # print(" Enter query ")
    DocID_store=dict()  
    DocID_list=[]
    query=input("Enter query: ")
    # Query set contain the preprocessing query.
    query_set=mainobj.preprocessing(query)     
    #convert the list into the set.
    query_set=set(query_set)
    filepath_map={}
    filename_map={}
    # traversing the file with their paths
    for filepath in glob.glob(directorypath, recursive=True):
        # print(filepath)
        if (path.isfile(filepath) and not filepath.endswith('.html')):
            # Calling the function which extract filename from path of a file and then store in a docname list
            filepath_map[mainobj.   docID]=filepath
            # Reading of file
            # Containing the file name
            filename_map[mainobj.docID]=mainobj.filename(filepath)
            with open(filepath, 'rb') as file:
                filecontent = file.read().decode(errors='replace')

            preprocessresult = []
            # Calling the function of preprocessing of file content
            #Preprocessing of the file content.
            preprocessresult = mainobj.preprocessing(filecontent)
            # print(preprocessresult)
            # print("done")
            #converting the list preprocessresult into the set.
            docset=set(preprocessresult)
            # union between doc and query
            union=docset.union(query_set)
            # intersection between doc and query

            intersection=docset.intersection(query_set)
            # finding the jaccard coefficient
            jaccard=len(intersection)/len(union)
            #Storing the jaccard coefficient into DocID_list
            DocID_list.append(jaccard)
            #This dictionary contains jaccard coefficient of each Documents with their doc_ID
            DocID_store[mainobj.docID]=jaccard
            #incresing the Doc_ID.
            mainobj.docID = mainobj.docID +1
    # print(DocID_store) 
    answerlist=[]
    # FInding the top 5 Documents.
    for i in range(5):
        maxakey=next(iter(DocID_store))
        for i in DocID_store.keys():
            if(DocID_store[i]>DocID_store[maxakey]):
                maxakey=i
        print(" The jacccard coefficient of top 5 Documents are ")

        print(DocID_store[maxakey])
        DocID_store.pop(maxakey)
        answerlist.append(maxakey)
    # print(" The jacccard coefficient of each Documents are ")
    print(answerlist)
    for i in answerlist:
        print(" The Document ID coefficient of each Documents are ")
        print(i)
        print("The File paths  of top 5 Documents.")
        print(filepath_map[i])
        print("The File Names  of top 5 Documents.")

        print(filename_map[i])

