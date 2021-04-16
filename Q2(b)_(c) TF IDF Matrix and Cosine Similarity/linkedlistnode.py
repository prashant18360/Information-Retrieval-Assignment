'''Creating the Node class for each node of a Linked list'''
class Node:
        #Attributes are IDval for storing the docID number, and next for storing the next node address
        def __init__(self, IDval=None):
                self.IDval = IDval
                self.next = None

'''Creating the LinkedList class for maintaining the each unique term's posting list'''
class Linkedlist:
        #Attributes are term for unique word, freq for size of linked list, headptr for pointing the starting node of linked list 
        def __init__(self, termid=None):
                self.termid = termid
                self.freq = 0
                self.headptr = None
                self.lastptr = None

        #Function for adding node in a linkedlist
        def addnode(self, IDdata):
                newNode = Node(IDdata)
                #Check that is it is a Empty linked list, so newNode make as a first node of a linked list
                if(self.headptr is None):
                        self.headptr = newNode
                        self.lastptr = newNode
                        self.freq = self.freq + 1
                        return

                #the last node of a non-empty linked list is in lastptr and add the newNode in the last and update the lastptr
                temp = self.lastptr
                temp.next = newNode
                self.lastptr = newNode
                self.freq = self.freq + 1
