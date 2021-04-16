import glob
import os.path
from os import path
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from linkedlistnode import Linkedlist
from os.path import split
from os.path import basename
import copy
import math
import numpy as np
import ntpath
import pickle
import sys
#this is only for serializing the class object by pickle dump 
sys.setrecursionlimit(10000)



class Mainfun:
    '''Attribute are unigram data structure for postinglist which is dictionary type data structure which is used for
    each unique words and their linkedlist containing all the docID in which word are present,
    docname which is a list of all document names, and docID maintaining the ID of document'''  
    def __init__(self):
        self.vocabdata = {}
        self.term_frequency_dict = {} #nested dictionary
        self.term_dict_id = {}
        self.termID = 0
        self.docID = 0
        self.docname = []
        self.temptermid = 0 #store termid in postinglist header
        self.dfdocfreq = {} #df for idf

        self.rows = 0
        self.cols = 0

        #Matrices for TF-IDF for 5 different schemes
        self.binary_Mat = []
        self.rawcount_Mat = []
        self.Termfrequency_Mat = []
        self.LogNormalisation_Mat = []
        self.DoubleNormalisation_Mat = []



    '''function  for preprocessing of a document content including converting into
    lower letter, remove punctuation, tokenization, remove stopping words and Lemmatization'''
    def preprocessing(self, content):
        #normalisation
        result1 = content.lower()
        result2 = result1.translate(str.maketrans("","", string.punctuation))

        #tokenization
        tokens = word_tokenize(result2)

        #removing the stopping words
        stop_words = set(stopwords.words('english'))
        result3 = [w for w in tokens if w not in stop_words]

        #Lemmatization
        lem = WordNetLemmatizer()
        result4dict = {}

        docvocabcount = {} #use for raw count
        for word in result3:
            lmword = lem.lemmatize(word)

            docvacabcount = self.term_rawcount(lmword, docvocabcount) #rawcount
            
            #here just only store the unique words of a file, 
            if(lmword not in result4dict):
                result4dict[lmword] = self.docID
        
        #store rawcount of word in a dictionary of a particular docID
        self.term_frequency_dict[self.docID] = docvocabcount 

        #now returning the result4dict, it is a dictionary type which contain all words from all document and their docID (key is words, value is docID)
        return(result4dict)




    '''Function for creating the unigram data structure containing all unique words and their linked list containing docID'''
    def Unigraminvertedindex(self, resultlist):
        
        #iterate all words in a resultlist
        for term in resultlist:
            
            #checking the particular words is present or not in a unigram data structure (postling list)
            if(term not in self.vocabdata):
                
                #Here word is not present then creating the new linked list of a new entry
                linkedlistobject = Linkedlist(self.temptermid)

                #adding the docID in a linkedlist
                linkedlistobject.addnode(resultlist.get(term))#resultlist.get(term) will give the docID

                #storing the word and linkedlist in a unigram data structure
                self.vocabdata[term] = linkedlistobject

                self.temptermid = self.temptermid + 1
                
            else:
                #Here word is present then retrieving the particular word's linkedlist
                tempobject = self.vocabdata.get(term)
                #adding the docID in a linkedlist
                tempobject.addnode(resultlist.get(term))



    '''function for counting the raw count of term in a particular document and store in a dictionary'''
    def term_rawcount(self, term, docvocabcount):
        
        #if term is not present then make a new entry and initialize count as 1 
        if term not in docvocabcount:
            docvocabcount[term] = 1

        #if term is present then pick the count value and increament by 1 
        else:
            tpcounty = docvocabcount[term]
            tpcounty = tpcounty + 1
            docvocabcount[term] = tpcounty

        return docvocabcount



    '''funtion for return the count value of term in a document from nested dictionary'''
    def Utility_Rawcounttf(self, r_ind, c_ind):
        tempdict = self.term_frequency_dict[r_ind]
        fetch_term = self.term_dict_id[c_ind]
        tcount = tempdict[fetch_term]

        return tcount


    '''funtion for return the total count value of every term in a document from nested dictionary'''
    def Utility_TermFreqtf(self, r_ind):
        Totalcount = 0
        tempdict = self.term_frequency_dict[r_ind]
        #pick out the nested dictionay and traverse over values
        for freq in tempdict.values():
            Totalcount = Totalcount + freq

        return Totalcount


    '''funtion for return the maximum count value from every term in a document from nested dictionary'''
    def Utility_DoubleNorm(self, r_ind):
        Maxcount = 0
        tempdict = self.term_frequency_dict[r_ind]
        #pick out the nested dictionay and traverse over values
        for freq in tempdict.values():
            if(Maxcount < freq):
                Maxcount = freq

        return Maxcount

    '''function for return the document frequency(how many document that term occurs) value of a term'''
    def Utility_idf_df(self, c_ind):
        term = self.term_dict_id[c_ind]
        templinkobj = self.vocabdata.get(term)
        return templinkobj.freq


    '''Funtion for creating the TF valued matrix of (0,1) weighting
    if term present in a particular document then marked as 1 , otherwise 0'''
    def Binary_tf(self, bintfMat):
        #iterate over every term from vocabulary 
        for term in self.vocabdata:
            self.term_dict_id[self.termID] = term

            #pick out the head pointer of posting list of that particular term
            templinkobj = self.vocabdata.get(term)
            tempptr = templinkobj.headptr
            #iterate posting list
            while tempptr is not None:
                tpdid = tempptr.IDval
                bintfMat[tpdid][self.termID] = 1
                tempptr = tempptr.next

            self.termID = self.termID + 1

        #stored a copy of matrix in a class attributes 
        self.binary_Mat = copy.copy(bintfMat)
        return bintfMat



    '''Funtion for creating the TF valued matrix of Rawcount weighting
        stored term count in a particular document '''
    def Rawcount_tf(self, rctfMat):
        for i in range(self.rows):
            for j in range(self.cols):
                if(rctfMat[i][j] == 1):
                    #call the function which will return the count value of that term
                    rctfMat[i][j] = self.Utility_Rawcounttf(i, j)

        #stored a copy of matrix in a class attributes
        self.rawcount_Mat = copy.copy(rctfMat)
        return rctfMat



    '''Function for creating the TF valued matrix of Termfrequency (rawcount of term/totalcount of term in document) weighting'''
    def TermFrequency_tf(self, tfMat):
        for i in range(self.rows):
            #call the func for return the total count of term in a document
            Totalnum = self.Utility_TermFreqtf(i)
            for j in range(self.cols):
                if(tfMat[i][j] > 0):
                    #store the value = rawcount of term/totalcount of term in document
                    tfMat[i][j] = tfMat[i][j]/Totalnum

        #stored a copy of matrix in a class attributes
        self.Termfrequency_Mat = copy.copy(tfMat)
        return tfMat



    '''Function for creating the TF valued matrix of Log Normalisation (log(1 + rawcount of term)) weighting'''
    def LogNormalisation_tf(self, tfMat):
        for i in range(self.rows):
            for j in range(self.cols):
                tfMat[i][j] = math.log(1+tfMat[i][j],10)

        #stored a copy of matrix in a class attributes
        self.LogNormalisation_Mat = copy.copy(tfMat)
        return tfMat



    '''Function for creating the TF valued matrix of Double Normalisation (0.5 + 0.5*rawcount of term/max rawcount from every term in a document) weighting'''
    def DoubleNormalisation_tf(self, tfMat):
        for i in range(self.rows):
            #call the func for return the max count from every term in a document
            Maxnum = self.Utility_DoubleNorm(i)
            for j in range(self.cols):
                tfMat[i][j] = 0.5 + (0.5*tfMat[i][j])/Maxnum

        #stored a copy of matrix in a class attributes
        self.DoubleNormalisation_Mat = copy.copy(tfMat)
        return tfMat


    '''Function for maultiply the IDF value with TF value in a matricex of 5 different schemes'''
    def MultiplyTF_idf(self):
        N = self.docID
        for j in range(self.cols):
            #call function which will return Document Frequency (df) value oa particular term
            df = self.Utility_idf_df(j)
            self.dfdocfreq[j] = df
            idf = math.log(N/(df+1), 10)  #calculating IDF value
            for i in range(self.rows):
                #Multiply all TF value with IDF value over all 5 different schemes and stored value in a respective matrices
                self.binary_Mat[i][j] = self.binary_Mat[i][j]*idf
                self.rawcount_Mat[i][j] = self.rawcount_Mat[i][j]*idf
                self.Termfrequency_Mat[i][j] = self.Termfrequency_Mat[i][j]*idf
                self.LogNormalisation_Mat[i][j] = self.LogNormalisation_Mat[i][j]*idf
                self.DoubleNormalisation_Mat[i][j] = self.DoubleNormalisation_Mat[i][j]*idf



    #Extracting the Document name from path of a document
    def fnamebypath(self, fpath):
        first, last = ntpath.split(fpath)
        return last or ntpath.basename(first)



if __name__ == '__main__':
    # set the directory's path of dataset
    directorypath = 'D:\IIITD\SEMESTER 6\Information Retrieval\Assignment\A2\stories\**\*'

    #creating the class object
    mainobj = Mainfun()
    
    #Here golb is used for the finding the all file path in a directory and also the inner directory file path
    for filepath in glob.glob(directorypath, recursive=True):
        if (path.isfile(filepath) and not filepath.endswith('.html')):

            #Calling the function which extract filename from path of a file and then store in a docname list 
            filename = mainobj.fnamebypath(filepath)
            mainobj.docname.append(filename)

            #Reading of file
            with open(filepath, 'rb') as file:
                filecontent = file.read().decode(errors='replace')
            
            preprocessresult = {}
            #Calling the function of preprocessing of file content
            preprocessresult = mainobj.preprocessing(filecontent)

            #Calling the function of creating the unigram data structure of each unique words of all file
            mainobj.Unigraminvertedindex(preprocessresult)

            mainobj.docID = mainobj.docID+1


    mainobj.rows, mainobj.cols = (mainobj.docID, len(mainobj.vocabdata))
    tfMat1 = np.zeros([mainobj.rows, mainobj.cols]) #initializing matrix of value 0 with size (No. of document x size of vocabulary)

    #Call the funtion which will return TF valued matrix of 5 different schemes
    tfMat2 = copy.copy(mainobj.Binary_tf(tfMat1))                   #Binary
    tfMat3 = copy.copy(mainobj.Rawcount_tf(tfMat2))                 #Rawcount
    tfMat4 = copy.copy(mainobj.TermFrequency_tf(tfMat3))            #TermFrequency
    tfMat5 = copy.copy(mainobj.LogNormalisation_tf(tfMat3))         #LogNormalisation
    tfMat6 = copy.copy(mainobj.DoubleNormalisation_tf(tfMat3))      #DoubleNormalisation


    #calling the function which will multiply TF with IDF value and store value in a respective matrices
    mainobj.MultiplyTF_idf()


    #Serialising the class object(in which all attribute like matrix, vocab stores) into store.dat file
    with open('store.dat' , 'wb') as fp:
        pickle.dump(mainobj, fp)




