from pandas import DataFrame
from pandas import Series
import numpy as np
from .Format import *

class Constraint():
    # unused
    def __init__(self):
        self.variables = variables  # 1d dataFrame
        self.sign = sign            # String
        self.RHS = RHS              # number

    @classmethod
    def fromArray(cls,variables,sign,RHS,names=False):
        if not names:
            names = [f"X{subscript(i+1)}"for i in range(variables)]
        variables = DataFrame(variables,columns=names)
        return Constraint(variables,sign,RHS)


class LP():
    @classmethod # raw constructor
    def new(cls,objectiveFn,constraints,signs,factorNames = False):
        #throw in some error checking to ensure valid LP
        objectiveType = objectiveFn[0]
        if not factorNames:
            factorNames = [f"X{subscript(i+1)}" for i in range(len(objectiveFn)-1)]
        factorNames = factorNames + ["signs","RHS"]
        body = DataFrame(
            [objectiveFn[1:]+["",""]]+
            constraints,
            columns=factorNames)
        return LP(objectiveType,body,signs)

    def __init__(self,objectiveType,body,signs):
        self.objectiveType = objectiveType  # String
        self.body = body                    # DataFrame
        self.signs = signs                  # List

    def getDual(self,factorPrefix = "Y",factorNames=[]):
        #factorNames is some List /nparray?

        if self.objectiveType == "min":
            constraint_sign  = {">":">","=":"urs","<":"<"}
            variable_sign    = {">":"<","urs":"=","<":">"}
        else:
            constraint_sign  = {"<":">","=":"urs",">":"<"}
            variable_sign    = {">":">","urs":"=","<":"<"}
        objectiveType = "min"if self.objectiveType == "max" else "max"
        
        body = self.body.copy()
        signs = self.signs.copy()

        obj = body.iloc[:,-1:].T.drop(0,axis=1)
        mid = body.iloc[:,:-2].drop(0).T
        obj = obj.append(mid).reset_index().drop("index",axis=1)
        
        if len(factorNames)== len(obj.columns):
            obj.columns = factorNames
        else:
            obj.columns = [f"{factorPrefix}{subscript(i)}" for i in obj.columns]

        obj.insert(obj.shape[1],"signs",np.array([""] + [variable_sign[i] for i in signs]))
        obj.insert(obj.shape[1],"RHS",np.append([""],body.iloc[0,:-2]))
        
        newSigns = list(body.loc[:,"signs"])[1:]
        newSigns = [constraint_sign[i] for i in newSigns]
        return LP(objectiveType,obj,newSigns)

    def getStandardForm(self):
        #set positive signs
        posSigns = self.posSigns()
        #modify urs factors
        noURS = posSigns.clearURS()
        #add surplus/slack variables
        SFactors = noURS.addSFactors()

        #set positive RHS
        return SFactors.posRHS()
    def posSigns(self):
        '''Set the X<0 factors to X>0'''
        objectiveType = self.objectiveType
        body = self.body.copy()
        signs = self.signs.copy()
        for i in range(len(signs)):
            if signs[i] == "<":
                signs[i] = ">"
                body.iloc[:,i] = body.iloc[:,i].apply(lambda x: x*-1)
        return LP(objectiveType,body,signs)
    def clearURS(self):
        objectiveType = self.objectiveType
        body = self.body.copy()
        signs = self.signs.copy()
        i = 0
        while i < len(signs):
            if signs[i] == "urs":
                signs[i] = ">"
                signs.insert(i,">")
                body.insert(i+1,body.columns[i]+"'",body.iloc[:,i])
                body.insert(i+2,body.columns[i]+"\"",body.iloc[:,i]*-1)
                body = body.drop(body.columns[i],axis=1)
            i+=1
        return LP(objectiveType,body,signs)
    def addSFactors(self):
        objectiveType = self.objectiveType
        body = self.body.copy()
        signs = self.signs.copy()
        for i in range(1,body.shape[0]):
            if body.iloc[i].loc["signs"] == "<":
                newCol = [0] * body.shape[0]
                newCol[i] = 1
                body.insert(body.shape[1]-2,f"s{subscript(i)}",np.array(newCol))
                signs.insert(body.shape[1]-2,">")
            elif body.iloc[i].loc["signs"] == ">":
                newCol = [0] * body.shape[0]
                newCol[i] = -1
                body.insert(body.shape[1]-2,f"s{subscript(i)}",np.array(newCol))
                signs.insert(body.shape[1]-2,">")
            body.iloc[i] = body.iloc[i].apply(lambda x: "=" if x in [">","<"] else x)
        return LP(objectiveType,body,signs)
    def posRHS(self):
        objectiveType = self.objectiveType
        body = self.body.copy()
        signs = self.signs.copy()
        for i in range(1,body.shape[0]):
            row = body.iloc[i]
            if row.loc["RHS"]<=0:
                body.iloc[i] = body.iloc[i].apply(lambda x: x*-1 if type(x)!=str else x)
        return LP(objectiveType,body,signs)

    def __eq__(self,other):
        if all(self.body.eq (other.body)):
            return True
        else:
            return False

    def display(self):       
        outDF = self.body.copy()
        outDF.insert(0,"",[1]+[""]*(len(self.body)-1))
        outDF.iloc[0,0] = self.objectiveType
        for i in outDF.columns[1:-2]:
            outDF.loc[:,i] = (outDF.loc[:,i].apply(lambda x: "" if x==0 else x))
            outDF.loc[:,i] = (outDF.loc[:,i].apply(str))
            outDF.loc[:,i] = (outDF.loc[:,i].apply(lambda x: x if (len(x)==0 or x[0]=="-") else "+"+x))
            outDF.loc[:,i] = (outDF.loc[:,i].apply(lambda x: x if len(x)==0 else x+i))

        signsDF = DataFrame([[""] + self.signs+["",""]],columns = outDF.columns)
        for i in signsDF.columns[1:-2]:
            signsDF.loc[:,i] = (signsDF.loc[:,i].apply(lambda x: i + x + " 0" if x in [">","<"] else x))
            signsDF.loc[:,i] = (signsDF.loc[:,i].apply(lambda x: i + " " + x if x =="urs" else x))
        outDF = outDF.append(signsDF)
        outDF = outDF.reset_index().drop(["index"],axis=1)
        return outDF
    def __repr__(self):
        return str(self.display())
