# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class HaveTofind_max_lengthFirstError(Exception):
    message = "find_max_length function has to been called in order to call make_equal function"
    def __init__(self):
        super().__init__(self.message)

class Transformer:                  

    max_dict = {}
    def __init__(self):
        self.foundmostlength = False
        
    
    def currencychanger(self,data,col):
        """if 5.83 USD it returns 583"""
        data[col] = data[col].str.split(" ").apply(lambda x: x[0].replace(".",""))
    
    
    def simplify(self,data,col):
        
        data[col] = data[col].apply(str)
        data[col] = data[col].str.strip()
    def split_columns(self,data,col):
        """turning the unique values of the column or columns(col) to column of the pandas dataframe(data),and delete the column or columns(col) """
        vectorizer = TfidfVectorizer()
        if not isinstance(col,list):
            col = [col]
        for co in col:
            x = vectorizer.fit_transform(data[co])
            newx = x.toarray() 
            del data[co]
            for index,column in enumerate(vectorizer.get_feature_names()):
                data[column] = newx[:,index]
    
    def replace_to_last(self,data,col):
        """first deleting and then adding the column or columns to the end of the dataframe"""
        if not isinstance(col,list):
            col = [col]
        for co in col:
            x = data[co]
            del data[co]
            data[co] = x
    
    
    def turn_int(self,data,col):
        """turns the float values into integers that have same digits
        for regression
        algorithms,
        first you need to call find_max_length
        function with the column
        to turn the column """
        
        if not self.foundmostlength:
            raise HaveTofind_max_lengthFirstError
        if not isinstance(col,list):
            col = [col]
        for co in col:
            data[co] = data[co].apply(self.float_to_int_equal,args = (co,))
        
    
    def float_to_int_equal(self,a,col):
        """makes the foat integers,adds zeroes to the end of the
           value compared to the longest float value in the same
           column,to make it more easy use turn_int instead"""
        if not self.foundmostlength:
            raise HaveTofind_max_lengthFirstError
        a = str(a).replace(".","")
        zerocount = "0"*(self.max_dict[col]-len(a))
        a += zerocount
        return int(a)

    def find_max_length(self,data,col):
        """
        finds the longest float in the column and adds to the
        dictionary max_dict to use it later
        """
        self.foundmostlength = True
        if  not isinstance(col,list):
            col = [col]
        for co in col:
            a = {len(str(i).split(",")[1]) for i in data[co]}
            self.max_dict[co] = max(a)-1

    def splitarray(self,data,last_x = 1,keep = None):
        """
        splits the array to the target(y) and values(x)
        last_x = how much rows from the end you want to be in y 
        keep = if you want to keep a spesific member of the array
        example:
        data -> 10000 rows,10 columns
        x,y,keep = splitarray(data,last_x = 2,keep = 'last')
        x = 9999x8
        y = 9999x2
        keep = data's last row
        """
        array = data.values
        if keep == 'last':
            keeped = array[-1]
            array = array[:-1]
        elif isinstance(keep,int):
            keeped = array[keep]
            del array[array.index(keep)]
        else:
            keeped = None
        
        length = len(data.columns)
        x = array[:,:length-last_x]
        
        if last_x == 1:
            y = array[:,length-last_x]
        else:
            y = array[:,length-last_x:]
        if keep != None:
            return x,y,np.array([keeped])
        else:
            return x,y

    @staticmethod
    def results(x,y,model,methods = None):
        """
        does the train_test and fit,etc.
        model is a algorithm;methods are to be used and functions like
        accuracy_score,you have to pass the functions;
        x and y are the splitted values of the 
        array(values of the data)
        example = 
        results(x,y,RandomForestRegressor(n_estimatores = 250),methods = [accuracy_score,mean_absolute_error])
        """
        X_train,X_test,Y_train,Y_test = train_test_split(x,y)
        model.fit(X_train,Y_train)
        preds = model.predict(X_test)
        try:
            print("model score:",model.score(X_train,Y_train))
        except:
            pass
        if methods == None:
            print(mean_absolute_error(Y_test,preds))
        print("first 10 elements of truth value and prediction",Y_test[:10],preds[:10])
        if methods!=None:
            if not isinstance(methods,list):
                methods = [methods]
            for method in methods:
                try:
                    print(f"{method.__name__} : ",method(Y_test,preds))
                except Exception as err:
                    print(f"error while processing on {method.__name__},ERROR:{err}")








