from util import entropy, information_gain, partition_classes
import numpy as np
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        pass

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree

        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split split)


        best_ig = -1000
        best_attribute = 0
        best_split = 0

        if len(set(y)) == 1:
            self.tree['label'] = y[0]
            return self.tree['label']

        for i in X:
            for attribute in range(len(i)):
                val = i[attribute]
                x_l,x_r,y_l,y_r = partition_classes(X,y,attribute,val)
                info_gain = information_gain(y,[y_l,y_r])
                if info_gain > best_ig:
                    best_ig = info_gain
                    best_attribute = attribute
                    best_split = val

        x_left,x_right,y_left,y_right = [],[],[],[]
        for i in range(len(X)):
            if X[i][best_attribute] <= best_split:
                x_left.append(X[i])
                y_left.append(y[i])
            else:
                x_right.append(X[i])
                y_right.append(y[i])



        tree_left = DecisionTree()
        tree_right = DecisionTree()
        tree_left.learn(x_left, y_left)
        tree_right.learn(x_right, y_right)

        self.tree['best_attribute'] = best_attribute
        self.tree['split'] = best_split
        self.tree['right'] = tree_right
        self.tree['left'] = tree_left


        pass


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        tree = self.tree
        while 'split' in tree:
            best_attr = tree['best_attribute']
            best_split = tree['split']
            if record[best_attr] <= best_split:
                tree = tree['left'].tree
            else:
                tree = tree['right'].tree
        return tree['label']
        pass
