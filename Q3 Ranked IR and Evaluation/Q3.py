import os
import ast
import numpy as np
import pandas as pd
import json
from  matplotlib import pyplot as plt
import math
col=["ab", "cd"]
train_file = open('/home/sudhir/Documents/New folder/IR-assignment-2-data.txt','r')
# train_file = pd.read_csv('/home/sudhir/Documents/New folder/IR-assignment-2-data.txt',header=None,sep="\t")
# text_file contains the linewise data.
text_file=train_file.read().splitlines()
# print(text_file[0])
count=0
#relevence_Score : it contains the relevence_Score
relevence_Score=[]
# QID_list contains all the lines of the text file in the string format.
QID_list=[]


DCG_list=[]

discounted_gain=[]
discounted_gain_reverse=[]
# print(len(text_file))
#relevence_Score and QID_list extraction
for i in  text_file:
	if (i[2:7]=='qid:4' and i[7]==' '):
		# print("qid")
		QID_list.append(i)	
		relevence_Score.append(i[0])	
		count+=1

# print(count)

# Here the conting the frequency of the relevence score in the relevence_Score_count_dict
print("relevence_Score ",relevence_Score)
relevence_Score_count_dict={}
relevence_Score_count_dict[0]=relevence_Score.count('0')
relevence_Score_count_dict[1]=relevence_Score.count('1')
relevence_Score_count_dict[2]=relevence_Score.count('2')
relevence_Score_count_dict[3]=relevence_Score.count('3')
print("Number of files possible are :--")
print(relevence_Score_count_dict)
# relevence_Score=[3,2,3,0,0,1,2,2,3,0]

# finding factorial
multiply_factorial=1
for i in relevence_Score_count_dict.keys():

	a=relevence_Score_count_dict[i]
	factorial_a=math.factorial(a)
	multiply_factorial*=factorial_a
print(multiply_factorial)
#   relevence_Score_sorted_reverse_list contains the relevence score in descending order.
relevence_Score_sorted_reverse_list=list(relevence_Score)
relevence_Score_sorted_reverse_list.sort(reverse=True)

# print("relevence_Score", relevence_Score_)
print("relevence_Score_sorted_reverse_list", relevence_Score_sorted_reverse_list)

#IDCG finding
discounted_gain_reverse.append(int(relevence_Score_sorted_reverse_list[0]))
for i in range(1,len(relevence_Score_sorted_reverse_list)):
	discounted_gain_reverse.append(float(relevence_Score_sorted_reverse_list[i])/np.log2(i+1))

print("discounted_gain_reverse")
print(discounted_gain_reverse)

# Writing in the text file.
output=open("Output.txt",'w')
output.close()
output=open("Output.txt",'a')
for i in QID_list:
	if (i[0]=='3'):
		output.write(i)
		output.write('\n')
		output.write('\n')
output.write('\n')
output.write('\n')
for i in QID_list:
	if (i[0]=='2'):
		output.write(i)
		output.write('\n')
		output.write('\n')
output.write('\n')
output.write('\n')
for i in QID_list:
	if (i[0]=='1'):
		output.write(i)
		output.write('\n')
		output.write('\n')
output.write('\n')
output.write('\n')
for i in QID_list:
	if (i[0]=='0'):
		output.write(i)
		output.write('\n')
		output.write('\n')
output.write('\n')
output.write('\n')

# DCG_list.append(discounted_gain_reverse[0])

#discounted gain by slide method DCG finding

discounted_gain.append(int(relevence_Score[0]))
for i in range(1,len(relevence_Score)):
	discounted_gain.append(float(relevence_Score[i])/np.log2(i+1))
print("discounted_gain")
print(discounted_gain)
DCG_list.append(discounted_gain[0])





# ////////\/\/\/\//\/\//\//Q2 from here//\/\/\/\//\/\/\/\//\/\//\//////////////////////////////////////////Q2 from here//\/\/\/\//\/\//\//\/\/\/\//\/\//\//\/\/\/\//\/\//\//\/\/\/\//\/\//\//\/\/\/\//\/\//\//\/\/\/\//\/\//\//\/\/\/\//\/\//\//\/\/\/\//\/\//\//\/\/\/\//\/\//\/
sum1=sum2=0


for i in range(50):
	sum1+=discounted_gain[i]
	sum2+=discounted_gain_reverse[i]
print("nDCG at 50 is ")
print(sum1/sum2)
print("nDCG of whole Documents")
# print(sum(discounted_gain/_reverse))
print(sum(discounted_gain)/sum(discounted_gain_reverse))


for i in range(1,len(discounted_gain)):

	DCG_list.append(DCG_list[i-1]+discounted_gain[i])

# print("QID_list",QID_list)






text_word_list=[]
text_75_word_list=[]
print("DCG")
print(DCG_list)


print(sum(discounted_gain))
print("Maximum DCG possible:-- ")
print(sum(discounted_gain_reverse))







# here converting the text of list into string format of word
for i in QID_list:
	text_word_list.append(i.split())





# /////////////////////////75 ////////////////////
#extraxting the 75th index
for i in range(len(text_word_list)):
	if (int(text_word_list[i][0])>0):
		text_75_word_list.append(float(text_word_list[i][76][3:]))
print(text_75_word_list)
print(len(text_75_word_list))
text_75_word_list_sorted=list(text_75_word_list)

text_75_word_list_sorted.sort(reverse=True)
Yes_NO=[]
for i in text_75_word_list_sorted:
	if i>=0:
		Yes_NO.append(1)
	else:
		Yes_NO.append(0)
Y=[]
N=[]
print(Yes_NO)
count_Y=0
count_N=0
for i in range(len(Yes_NO)):
	if Yes_NO[i]==	1:
		count_Y+=1
	else:
		count_N+=1
	Y.append(count_Y)
	N.append(count_N)
print(Y)
print(N)
Precision=[]
recall=[]
for i in range(len(Yes_NO)):
	Precision.append(Y[i]/(Y[i]+N[i]))
	recall.append(Y[i]/len(Yes_NO))
print("Precision",Precision)
print(recall)


plt.plot(recall,Precision)
plt.show()

