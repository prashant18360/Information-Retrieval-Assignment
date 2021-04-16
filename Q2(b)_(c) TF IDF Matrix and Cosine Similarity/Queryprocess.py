from Tf_idf import Mainfun
from linkedlistnode import Linkedlist
import string
import math
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import copy
import pickle




class QueryProcess:
    def __init__(self):
        '''Attribute for store the final file name of top 5 most relevant document in all 5 different schemes of TF-IDF Score and Cosine similarity'''
        self.fnamelist_binary = []
        self.fnamelist_rawcount = []
        self.fnamelist_termfreq = []
        self.fnamelist_lognormal = []
        self.fnamelist_doublenormal = []

        self.cosine_fnamelist_binary = []
        self.cosine_fnamelist_rawcount = []
        self.cosine_fnamelist_termfreq = []
        self.cosine_fnamelist_lognormal = []
        self.cosine_fnamelist_doublenormal = []



    '''function  for preprocessing of a query including converting into
    lower letter, remove punctuation, tokenization, remove stopping words and Lemmatization'''
    def preprocess(self, query):
        #normalisation
        result1 = query.lower()
        result2 = result1.translate(str.maketrans("","", string.punctuation))

        #tokenization
        tokens = word_tokenize(result2)
        
        #removing the stopping words
        stop_words = set(stopwords.words('english'))
        result3 = [w for w in tokens if w not in stop_words]

        #Lemmatization
        lem = WordNetLemmatizer()
        result4query = []
        for word in result3:
            lmword = lem.lemmatize(word)
            result4query.append(lmword)

        return(result4query)


    '''function for creating the query vector after preprocessed of query string'''
    def CreateQvector(self, prepresult, dictlist, cols):
        #initialize the query vector of vocab size
        qvector = np.zeros((cols,), dtype=int)
        
        extra_dict_temp = {}
        j=0

        for term in prepresult:
            if term in dictlist:
                templinkobj = dictlist[term]
                #count of term in query
                qvector[templinkobj.termid] = qvector[templinkobj.termid] + 1
            #if term is a not lie in document vocab, then add the new entry
            else:
                if term in extra_dict_temp:
                    temp_ind = extra_dict_temp[term]
                    #count of term in query
                    qvector[temp_ind] = qvector[temp_ind] + 1
                else:
                    qvector.append(1)
                    extra_dict_temp[term] = cols + j
                    j=j+1
        return qvector


    '''function for score finding of every document according to the query term lie in it or not and store the top 5 most relevant docname'''
    def TFidf_Scorefinding(self, qvector, mat1, mat2, mat3, mat4, mat5, rows, cols, fnamelist):
        queryscore1 = {}
        queryscore2 = {}
        queryscore3 = {}
        queryscore4 = {}
        queryscore5 = {}
        for i in range(rows):
            score1 = 0
            score2 = 0
            score3 = 0
            score4 = 0
            score5 = 0
            for j in range(cols):
                if qvector[j] >= 1:
                    #calculating of score of all 5 matrix
                    score1 = score1 + mat1[i][j]
                    score2 = score2 + mat2[i][j]
                    score3 = score3 + mat3[i][j]
                    score4 = score4 + mat4[i][j]
                    score5 = score5 + mat5[i][j]
            #store score with document ID
            queryscore1[i] = score1
            queryscore2[i] = score2
            queryscore3[i] = score3
            queryscore4[i] = score4
            queryscore5[i] = score5

        #Sort all 5 score data structure in a Descending order
        sorted_score1 = sorted(queryscore1, key=queryscore1.get, reverse=True)
        sorted_score2 = sorted(queryscore2, key=queryscore2.get, reverse=True)
        sorted_score3 = sorted(queryscore3, key=queryscore3.get, reverse=True)
        sorted_score4 = sorted(queryscore4, key=queryscore4.get, reverse=True)
        sorted_score5 = sorted(queryscore5, key=queryscore5.get, reverse=True)


        #Retrieve top 5 docID, and pick out the docname by docID and store into the class attributes
        for i in range(5):
            self.fnamelist_binary.append(fnamelist[sorted_score1[i]])
            self.fnamelist_rawcount.append(fnamelist[sorted_score2[i]])
            self.fnamelist_termfreq.append(fnamelist[sorted_score3[i]])
            self.fnamelist_lognormal.append(fnamelist[sorted_score4[i]])
            self.fnamelist_doublenormal.append(fnamelist[sorted_score5[i]])



    '''Function for calculating the TF-IDF value of query vector of all 5 different schemes'''
    def queryTF_IDF(self, Queryvector, termdf):
        
        qvect_tfidf_binary = []
        qvect_tfidf_lognorm = []
        qvect_tfidf_doubnorm = []

        vectsize = len(Queryvector)
        vecttm_count = sum(Queryvector)
        Maxnum = max(Queryvector)

        #calculation of TF
        for i in range(vectsize):
            tf1 = math.log(1+Queryvector[i],10)
            qvect_tfidf_lognorm.append(tf1)     #LogNormal
            tf2 = 0.5 + (0.5*Queryvector[i])/Maxnum
            qvect_tfidf_doubnorm.append(tf2)    #Double Normal
            #Binary
            if(Queryvector[i] > 0):
                qvect_tfidf_binary.append(1)
            else:
                qvect_tfidf_binary.append(0)

        qvect_tfidf_rawcount = copy.copy(Queryvector)   #RawCount
        qvect_tfidf_termfreq = copy.copy(Queryvector)
        qvect_tfidf_termfreq = qvect_tfidf_termfreq/vecttm_count    #TermFrequency

        #calculating TF-IDF and store in a qvector
        for i in range(vectsize):
            dfi = termdf.get(i)
            qvect_tfidf_binary[i] = qvect_tfidf_binary[i]*dfi   #binary
            qvect_tfidf_rawcount[i] = qvect_tfidf_rawcount[i]*dfi  #rawcount
            qvect_tfidf_termfreq[i] = qvect_tfidf_termfreq[i]*dfi  #TermFrequency
            qvect_tfidf_lognorm[i] = qvect_tfidf_lognorm[i]*dfi   #Log Normalisation
            qvect_tfidf_doubnorm[i] = qvect_tfidf_doubnorm[i]*dfi  #Double Normalisation

        #return query vector (of all 5 different schemes) of TF-IDF score
        return np.array(qvect_tfidf_binary), np.array(qvect_tfidf_rawcount), np.array(qvect_tfidf_termfreq), np.array(qvect_tfidf_lognorm), np.array(qvect_tfidf_doubnorm)
            
            
        

    '''Function for calculating the cosine similarity between documaent and query in all 5 different schemes'''
    def cosinesimilarity(self, Queryvector, termdf, mat1, mat2, mat3, mat4, mat5, rows, cols, fnamelist):

        #Calling the Function for calculating the TF-IDF value of query vector of all 5 different schemes
        qvect1, qvect2, qvect3, qvect4, qvect5 = self.queryTF_IDF(Queryvector, termdf)

        queryscore1 = {}
        queryscore2 = {}
        queryscore3 = {}
        queryscore4 = {}
        queryscore5 = {}

        vectlen = len(qvect1)
        
        #calculating the magnitude of query vector
        Qmod1 = np.sqrt(qvect1.dot(qvect1))
        Qmod2 = np.sqrt(qvect2.dot(qvect2))
        Qmod3 = np.sqrt(qvect3.dot(qvect3))
        Qmod4 = np.sqrt(qvect4.dot(qvect4))
        Qmod5 = np.sqrt(qvect5.dot(qvect5))

        
        for i in range(rows):
            #Calculating the dot product of document and query vector
            QD1dotprod = np.dot(mat1[i], qvect1[:cols])
            QD2dotprod = np.dot(mat2[i], qvect2[:cols])
            QD3dotprod = np.dot(mat3[i], qvect3[:cols])
            QD4dotprod = np.dot(mat4[i], qvect4[:cols])
            QD5dotprod = np.dot(mat5[i], qvect5[:cols])
            
            for j in range(cols, vectlen):
                QD5dotprod = QD5dotprod + (qvect5[j]*0.5)

            #Calculating the magnitude of document vector
            Docmod1 = np.sqrt(mat1[i].dot(mat1[i]))
            Docmod2 = np.sqrt(mat2[i].dot(mat2[i]))
            Docmod3 = np.sqrt(mat3[i].dot(mat3[i]))
            Docmod4 = np.sqrt(mat4[i].dot(mat4[i]))
            Docmod5 = np.sqrt(mat5[i].dot(mat5[i]))

            #Final cos score between document and query
            score1 = QD1dotprod/(Docmod1*Qmod1)
            score2 = QD2dotprod/(Docmod2*Qmod2)
            score3 = QD3dotprod/(Docmod3*Qmod3)
            score4 = QD4dotprod/(Docmod4*Qmod4)
            score5 = QD5dotprod/(Docmod5*Qmod5)

            #Store the document ID with corresponding to their cosine similarity score 
            queryscore1[i] = score1
            queryscore2[i] = score2
            queryscore3[i] = score3
            queryscore4[i] = score4
            queryscore5[i] = score5


        #Sort all 5 score data structure in a Descending order
        sorted_score1 = sorted(queryscore1, key=queryscore1.get, reverse=True)
        sorted_score2 = sorted(queryscore2, key=queryscore2.get, reverse=True)
        sorted_score3 = sorted(queryscore3, key=queryscore3.get, reverse=True)
        sorted_score4 = sorted(queryscore4, key=queryscore4.get, reverse=True)
        sorted_score5 = sorted(queryscore5, key=queryscore5.get, reverse=True)


        ##Retrieve top 5 docID, and pick out the docname by docID and store into the class attributes
        for i in range(5):
            self.cosine_fnamelist_binary.append(fnamelist[sorted_score1[i]])
            self.cosine_fnamelist_rawcount.append(fnamelist[sorted_score2[i]])
            self.cosine_fnamelist_termfreq.append(fnamelist[sorted_score3[i]])
            self.cosine_fnamelist_lognormal.append(fnamelist[sorted_score4[i]])
            self.cosine_fnamelist_doublenormal.append(fnamelist[sorted_score5[i]])
                    






if __name__ == '__main__':
    #Deserailization of class object, in which all TF-idf matrix has stored
    with open('store.dat' , 'rb') as fr:
        tempomainobj = pickle.load(fr)
    
    #retriving the all TF-idf matrix from store.dat file
    binary_Mat = tempomainobj.binary_Mat
    rawcount_Mat = tempomainobj.rawcount_Mat
    Termfrequency_Mat = tempomainobj.Termfrequency_Mat
    LogNormalisation_Mat = tempomainobj.LogNormalisation_Mat
    DoubleNormalisation_Mat = tempomainobj.DoubleNormalisation_Mat

    rows = tempomainobj.rows
    cols = tempomainobj.cols

    #retriving the df,vocabulary,filename from store.dat file
    termdf = tempomainobj.dfdocfreq
    dictlist = tempomainobj.vocabdata
    filename = tempomainobj.docname
    
    #Input the no. of query from the User
    n = int(input("Enter the number of Query: "))
    
    for i in range(n):
        #input the query and query operator 
        query = input("Input Query: ")
        
        #Preprocessing of Query
        Queryobj = QueryProcess()
        prepresult = Queryobj.preprocess(query)

        #calling the funtion to Create the query vector from the query string after preprocessing 
        Queryvector = copy.copy(Queryobj.CreateQvector(prepresult, dictlist, cols))

        #Calling the function to score finding of every document and store the top 5 most relevant docname according to the query term lie in it or not
        Queryobj.TFidf_Scorefinding(Queryvector, binary_Mat, rawcount_Mat, Termfrequency_Mat, LogNormalisation_Mat, DoubleNormalisation_Mat, rows, cols, filename)
        

        #Print the Docname of all 5 different scemes
        print("Top 5 relevant document based on TF-IDF score:")
        print("Binary Weighting Scheme :- ", Queryobj.fnamelist_binary)
        print("Raw Count Weighting Scheme :- ", Queryobj.fnamelist_rawcount)
        print("Term Frequency Weighting Scheme :- ", Queryobj.fnamelist_termfreq)
        print("Log Normalisation Weighting Scheme :- ", Queryobj.fnamelist_lognormal)
        print("Double Normalisation Weighting Scheme :- ", Queryobj.fnamelist_doublenormal)
        print()
        


        #Calling the function to calculate the cosine similarity between every document with query and store the top 5 most relevant docname 
        Queryobj.cosinesimilarity(Queryvector, termdf, binary_Mat, rawcount_Mat, Termfrequency_Mat, LogNormalisation_Mat, DoubleNormalisation_Mat, rows, cols, filename)


        #Print the Docname of all 5 different scemes
        print("Top 5 relevant document based on Cosine Similarity :")
        print("Binary Weighting Scheme :- ", Queryobj.cosine_fnamelist_binary)
        print("Raw Count Weighting Scheme :- ", Queryobj.cosine_fnamelist_rawcount)
        print("Term Frequency Weighting Scheme :- ", Queryobj.cosine_fnamelist_termfreq)
        print("Log Normalisation Weighting Scheme :- ", Queryobj.cosine_fnamelist_lognormal)
        print("Double Normalisation Weighting Scheme :- ", Queryobj.cosine_fnamelist_doublenormal)
        print()
        print()

        

        
        



