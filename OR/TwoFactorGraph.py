import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_MinMaxY(myLimits):
    store = dict()
    for i in myLimits:
        if i[0] in store.keys():
            store[i[0]] +=[i]
        else:
            store[i[0]] =[i]
    minVals = []
    maxVals = []
    for k,v in store.items():
        maxY = max([i[1] for i in v])
        minY = min([i[1] for i in v])
        minVals.append((k,minY))
        maxVals.append((k,maxY))
    minVals.sort(key=lambda x:x[0])
    maxVals.sort(key=lambda x:x[0])
    return minVals,maxVals

def get_MinMaxX(myLimits):
    store = dict()
    for i in myLimits:
        if i[1] in store.keys():
            store[i[1]] +=[i]
        else:
            store[i[1]] =[i]
    minVals = []
    maxVals = []
    for k,v in store.items():
        maxX = max([i[0] for i in v])
        minX = min([i[0] for i in v])
        minVals.append((minX,k))
        maxVals.append((maxX,k))

    minVals.sort(key=lambda x:x[1])
    maxVals.sort(key=lambda x:x[1])
    return minVals,maxVals

def graphical_LP(constraints,xMax=10,yMax=10,integer=False,shading=True,objectiveFunction=False):
    nonegative = True
    left = 0
    bottom = 0
    right=xMax
    top=yMax
    if integer:
        yDomain = list(range(bottom,yMax+1)) #(bottom,yMax)
        xDomain = list(range(left,xMax+1)) #(left,xMax)
    else:
        yDomain = np.linspace(bottom,yMax,1000) #(bottom,yMax)
        xDomain = np.linspace(left,xMax,1000) #(left,xMax)

    fig,axes  = plt.subplots()
    feasible = set((x,y) for x in xDomain for y in yDomain)
    
    colors = plt.rcParams['axes.prop_cycle']()
    for constraint,color in zip(constraints,colors):
        myColor = color['color']
        if constraint[1] in (">",">=") :
            myLimits = set((x,y) for x in xDomain for y in yDomain if constraint[0](x,y) >= constraint[2] )
            feasible = feasible.intersection(myLimits)
        elif constraint[1] in ("<","<="):
            myLimits = set((x,y) for x in xDomain for y in yDomain if constraint[0](x,y) <= constraint[2] )
            feasible = feasible.intersection(myLimits)
        elif constraint[1] == "=":
            myLimits = set((x,y) for x in xDomain for y in yDomain if abs(constraint[0](x,y) - constraint[2])<0.1)
            feasible = feasible.intersection(myLimits)
        
        #feasible area for this constraint 
        minY,maxY = get_MinMaxY(myLimits)
        minX,maxX = get_MinMaxX(myLimits)
        axes.plot([i[0] for i in minY],[i[1] for i in minY],color=myColor)# ,label = constraint)
        axes.plot([i[0] for i in maxY],[i[1] for i in maxY],color=myColor)
        axes.plot([i[0] for i in minX],[i[1] for i in minX],color=myColor)
        axes.plot([i[0] for i in maxX],[i[1] for i in maxX],color=myColor)

        if shading:
            allVals = minY+maxX+list(reversed(maxY))+list(reversed(minX))
            axes.add_patch(Polygon(allVals,alpha=0.25/len(constraints),color=myColor))
    if integer:
        axes.scatter(
            [i[0] for i in feasible],
            [i[1] for i in feasible],
            color="green"
        )
    
    if shading:
        minY,maxY = get_MinMaxY(feasible)
        minX,maxX = get_MinMaxX(feasible)
        allVals = minY+maxX+list(reversed(maxY))+list(reversed(minX))
        if len(allVals)>0:
            axes.add_patch(Polygon(allVals,alpha=0.25,color="green"))



    # if objectiveFunction:
    #     MaxVal = 0
    #     MaxCoords = []
    #     for x,y in feasible:
    #         test = objectiveFunction(x,y)
    #         if test>MaxVal:
    #             MaxVal = test
    #             MaxCoords = [(x,y)]
    #         elif abs(test - MaxVal)<0.1:
    #             MaxCoords += [(x,y)]
    
    #     xPlot = [i[0] for i in MaxCoords]
    #     yPlot = [i[1] for i in MaxCoords]
    #     plt.plot(xPlot,yPlot)

    axes.set_ylim(bottom=bottom,top=top)
    axes.set_xlim(left=left,right=right)
    axes.set_xlabel(u"X\u2081")
    axes.set_ylabel(u"X\u2082")
    axes.grid()
    return fig,axes



# Sample
# hw3

# constraints = [
#     (lambda x1,x2 :x1+2*x2, "<",350),
#     (lambda x1,x2: 2*x1+x2, "<",400)
#     ]
# fig,axes = graphical_LP(constraints,xMax=220,yMax=200)
# axes.plot(-1,-1,color="#1f77b4",label = u"X\u2081+2X\u2082 ≤ 350")
# axes.plot(-1,-1,color="#ff7f0e",label = u"2X\u2081+X\u2082 ≤ 400")
# axes.legend()
# fig.show()
# input("exit")