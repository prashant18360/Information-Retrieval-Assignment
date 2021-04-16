import nltk
from nltk import PorterStemmer
from ass2 import mainfun
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

class QueryProcess:

    def preprocessing(self, content):
        content = content.lower()
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        content_tokens = tokenizer.tokenize(content)
        r_1 = []
        stop_words = set(stopwords.words('english'))
        for w in content_tokens:
            if w not in stop_words:
                r_1.append(w)

        r_2 = [t for t in r_1 if t not in string.punctuation]

        r_3 = []
        for d in r_2:
            if d != " ":
                r_3.append(d)
        stemmer = PorterStemmer()
        result = []
        for x in r_3:
            x = stemmer.stem(x)
            result.append(x)

        return result
    def searching(self, list1, query,docname):
        size=len(list1)
        l=[]
        #print(size)

        l1=list1[0][1]
        #checking common docs for the query in the list
        c_docs=[]

        for x in (l1):

            check = 0
            for i in range(1,size):
                if x in list1[i][1]:

                    check+=1

            if(check==size-1):
                 c_docs.append(x)

        #retrieving the list of positions for words from the common_docs
        b_list=[]
        for c in c_docs:

            l={}
            for i in range(size):
                if c not in l:
                    l[c]=[list1[i][1][c]]
                else:
                    l[c].append(list1[i][1][c])

            b_list.append(l)



        main_list=[]
        iter=0
        for v in b_list:
            for d in v:
                list1=v.get(d)[0]

                for val in list1:
                    bool = True
                    for x in range(1,len(v.get(d))):
                        val += 1
                        if val not in v.get(d)[x]:
                            bool=False
                            break
                    if (bool==True):
                        main_list.append(docname[d])
        main_dict=dict.fromkeys(main_list)
        main_l=[]
        for i in main_dict:
            main_l.append(i)
        #printing the main list containing all the files names
        print("Total No. of docs : ", len(main_l))
        print(main_l)




if __name__ == '__main__':
    with open('store.dat' , 'rb') as fr:
        temp_mainobj = pickle.load(fr)
    index = temp_mainobj.p_index
    docname=temp_mainobj.docname
    # input the query and query operator
    query = input("Input Query: ")
    # Preprocessing of Query
    Queryobj = QueryProcess()
    result = Queryobj.preprocessing(query)

    list=[]
    for x in result:
        l=index.get(x)
        list.append(l)


    Queryobj.searching(list,result,docname)
