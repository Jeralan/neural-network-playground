from cmu_112_graphics import *
from tkinter import *

class Playground(ModalApp):
    def appStarted(self):
        self.promptMode =  PromptMode()
        self.gameMode = GameMode()
        self.functionMode = FunctionMode()
        self.setActiveMode(self.promptMode)
        self.timerDelay = 1

def tupleReLU(L):
    L = list(L)
    for x in range(len(L)):
        if L[x] < 0: L[x] = 0
    return tuple(L)

#taken from https://www.cs.cmu.edu/~112/notes/notes-recursion-part2.html#memoization
def memoized(f):
    import functools
    cachedResults = dict()
    @functools.wraps(f)
    def wrapper(*args):
        if args not in cachedResults:
            cachedResults[args] = f(*args)

        return cachedResults[args]
    return wrapper

@memoized
def forward(x,weights,biases,depth):
    if len(weights) == 0:
        return x
    prevLayer = forward(x,weights[:-1],biases[:-1],depth-1)
    layerOutput = []
    for i in range(len(weights[-1])):
        neuron = 0
        for j in range(len(prevLayer)):
            neuron += prevLayer[j]*weights[-1][i][j]
        neuron += biases[-1][i]
        layerOutput.append(neuron)
    if depth == 0: return layerOutput
    else: return tupleReLU(layerOutput)

def tupleize(L):
    if type(L) != list:
        return L
    output = []
    for subList in L:
        output.append(tupleize(subList))
    return tuple(output)

def meanSquared(L1,L2):
    squares = [(L1[i]-L2[i])**2 for i in range(len(L1))]
    return sum(squares)/len(squares)

import copy

class GameMode(Mode):
    def appStarted(self):
        self.valErrors = []
        self.trainErrors = []
        self.hiddenSizes = self.app.hiddenSizes+[1]
        self.xTrain,self.xVal,self.xTest = self.app.xTrain,self.app.xVal,self.app.xTest
        self.yTrain,self.yVal,self.yTest = self.app.yTrain,self.app.yVal,self.app.yTest
        self.xMax,self.xMin = max(self.xTrain),min(self.xTrain)
        self.yMax,self.yMin = max(self.yTrain),min(self.yTrain)
        self.xTrain = [(x-self.xMin)/(self.xMax-self.xMin) for x in self.xTrain]
        self.xVal = [(x-self.xMin)/(self.xMax-self.xMin) for x in self.xVal]
        self.xTest = [(x-self.xMin)/(self.xMax-self.xMin) for x in self.xTest]
        if self.yMax != self.yMin:
            self.yTrain = [(y-self.yMin)/(self.yMax-self.yMin) for y in self.yTrain]
            self.yVal = [(y-self.yMin)/(self.yMax-self.yMin) for y in self.yVal]
            self.yTest = [(y-self.yMin)/(self.yMax-self.yMin) for y in self.yTest]
        else:
            self.yTrain = [y/self.yMax for y in self.yTrain]
            self.yVal = [y/self.yMax for y in self.yVal]
            self.yTest = [y/self.yMax for y in self.yTest]
        self.layers = len(self.hiddenSizes)
        self.timerDelay = 50
        self.currentLayer = 1
        self.forwardState = True
        self.weightChecking = None
        self.weights = []
        self.biases = []
        self.timer = 0
        for layer in range(self.layers):
            layerWeights = []
            layerBiases = []
            for neuron in range(self.hiddenSizes[layer]):
                if layer != 0:
                    neuronWeights = []
                    for neuron in range(self.hiddenSizes[layer-1]):
                        neuronWeights.append(random.randint(-100,100)/100)
                else:
                    neuronWeights = [random.randint(-100,100)/100]
                layerWeights.append(neuronWeights)
                layerBiases.append(0)
            self.weights.append(layerWeights)
            self.biases.append(layerBiases)
        #weights: input layer > input neuron > output neuron 

    def backward(self):
        newWeights = copy.deepcopy(self.weights)
        newBiases = copy.deepcopy(self.biases)
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += self.app.interval
                    yTrainPred = [self.forward([x],self.weights,self.biases)[0] for x in self.xTrain]
                    yTrainError = meanSquared(yTrainPred,self.yTrain)
                    newWeights[i][j][k] += (-(yTrainError-self.yTrainError)/self.app.interval)*self.app.learnRate
                    self.weights[i][j][k] -= self.app.interval
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                self.biases[i][j] += self.app.interval
                yTrainPred = [self.forward([x],self.weights,self.biases)[0] for x in self.xTrain]
                yTrainError = meanSquared(yTrainPred,self.yTrain)
                newBiases[i][j] += (-(yTrainError-self.yTrainError)/self.app.interval)*self.app.learnRate
                self.biases[i][j] -= self.app.interval
        return (newWeights,newBiases)
    
    def forward(self,x,weights,biases):
        return forward(tupleize(x),tupleize(weights),tupleize(biases),0)
        
    def timerFired(self):
        self.timer += 1
        yTrainPred = [self.forward([x],self.weights,self.biases)[0] for x in self.xTrain]
        yValPred = [self.forward([x],self.weights,self.biases)[0] for x in self.xVal]
        self.yTrainError = meanSquared(yTrainPred,self.yTrain)
        self.yValError = meanSquared(yValPred,self.yVal)
        self.valErrors.append(self.yValError)
        self.trainErrors.append(self.yTrainError)
        self.weights,self.biases = self.backward()

    def drawNeurons(self, canvas):
        layers = len(self.weights)+1
        spacing = self.networkArea/(2*layers+1)
        r = self.app.height/(max(self.hiddenSizes)*2+2)*2/3
        cx,cy = spacing+r,self.app.height/2
        canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="black")
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                cx,cy = spacing*3+spacing*2*i+r,self.app.height/(len(self.biases[i])+1)*(j+1)
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="black")
                if self.biases[i][j] >= 0:
                    text = f"+{round(self.biases[i][j],3)}"
                else:
                    text = f"{round(self.biases[i][j],3)}"
                canvas.create_text(cx,cy,text=text,fill="white",anchor="c",font=f"arial {int(r/2)}")

    def drawWeights(self, canvas):
        layerSizes = [1] + self.hiddenSizes
        layers = len(self.weights)+1
        spacing = self.networkArea/(2*layers+1)
        r = self.app.height/(max(self.hiddenSizes)*2+2)*2/3
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    width = abs(self.weights[i][j][k])*20
                    if self.weights[i][j][k] > 0: color = "red"
                    else: color = "blue"
                    startLayer,endLayer = i,i+1
                    x1,y1 = spacing*3+spacing*2*(i-1)+r,self.app.height/(len(self.biases[i-1])+1)*(k+1)
                    x2,y2 = spacing*3+spacing*2*i+r,self.app.height/(len(self.biases[i])+1)*(j+1)
                    canvas.create_line(x1,y1,x2,y2,width=width,fill=color)
    
    def drawError(self, canvas):
        errorWidth = self.app.width/3
        errorHeight = self.app.height/2
        points = min(len(self.valErrors),int(errorWidth))
        if points > 1:
            maxError = max(max(self.valErrors[0:points]),max(self.trainErrors[0:points]))
            r = 2
            maxHeight = 0
            for i in range(1,points+1):
                cx = self.app.width - errorWidth*(i/points)
                cy = errorHeight - (self.valErrors[-i]/maxError)*errorHeight + 10
                maxHeight = max(cy,maxHeight)
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="yellow")
                cy = errorHeight - (self.trainErrors[-i]/maxError)*errorHeight + 10
                maxHeight = max(cy,maxHeight)
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="black")
            cx = self.app.width-errorWidth
            cy = maxHeight
            canvas.create_text(cx,cy,anchor="ne",text=f"{min(min(self.valErrors[0:points]),min(self.trainErrors[0:points]))}")
            cy = 0
            canvas.create_text(cx,cy,anchor="ne",text=f"{maxError}")

    def redrawAll(self, canvas):
        self.networkArea = self.app.width*3/4
        self.drawWeights(canvas)
        self.drawNeurons(canvas)
        self.drawError(canvas)

import random
def createXList(left,right,num):
    xList = random.sample(range(left,right),num)
    return xList

class FunctionButton(object):
    def __init__(self,function,name,parends=False):
        self.function = function
        self.name = name
        self.parends = parends
    def applyFunction(self,functionString,displayString):
        if self.parends:
            return (self.function + "(" + functionString + ")",self.name + "(" + displayString + ")")
        else: return (functionString + self.function,displayString + self.name)
    def draw(self,canvas,x,y,h,w):
        canvas.create_rectangle(x,y,x+w,y+h,fill="white",width=2)
        if self.parends:
            canvas.create_text(x+w/2,y+h/2,text=f"{self.name}()")
        else:
            canvas.create_text(x+w/2,y+h/2,text=self.name)

import math
class FunctionMode(Mode):
    def appStarted(self):
        self.app.xList = None
        self.functionString = ""
        self.displayString = ""
        self.buttons = [FunctionButton("math.cos","Cos",parends=True),
                        FunctionButton("math.sin","Sin",parends=True),
                        FunctionButton("math.tan","Tan",parends=True),
                        FunctionButton("x","x"),
                        FunctionButton("*","*"),
                        FunctionButton("+","+"),
                        FunctionButton("**","**"),
                        FunctionButton("1","1"),
                        FunctionButton("2","2"),
                        FunctionButton("3","3"),
                        FunctionButton("4","4"),
                        FunctionButton("5","5"),
                        FunctionButton("6","6"),
                        FunctionButton("7","7"),
                        FunctionButton("8","8"),
                        FunctionButton("9","9"),
                        FunctionButton("0","0"),
                        FunctionButton("/","/")]
    def mousePressed(self,event):
        if self.app.xList == None:
            self.Left = int(self.getUserInput('Integer Left Bound of X?'))
            self.Right = int(self.getUserInput('Integer Right Bound of X?'))
            self.num = self.Right-self.Left
            self.app.xList = createXList(self.Left,self.Right,self.num)
            self.app.yList = self.app.xList
        for i in range(len(self.buttons)):
            if event.x > self.buttons[i].xs[0] and event.x < self.buttons[i].xs[1]:
                if event.y > self.buttons[i].ys[0] and event.y < self.buttons[i].ys[1]:
                    self.functionString,self.displayString = self.buttons[i].applyFunction(self.functionString,self.displayString)
    def keyPressed(self,event):
        if event.key == "d":
            yList = []
            for element in self.app.xList:
                x = element
                yList.append(eval(self.functionString))
            self.app.yList = yList
            xList = self.app.xList
            testSplit = int(self.app.testSplit*len(xList))
            xTest = xList[:testSplit]
            yTest = yList[:testSplit]
            self.app.xTest,self.app.yTest = xTest,yTest
            xList = xList[testSplit:]
            yList = yList[testSplit:]
            valSplit = int(self.app.valSplit*len(xList))
            xVal = xList[:valSplit]
            yVal = yList[:valSplit]
            self.app.xVal,self.app.yVal = xVal,yVal
            xList = xList[valSplit:]
            yList = yList[valSplit:]
            self.app.xTrain,self.app.yTrain = xList,yList
            self.app.setActiveMode(self.app.gameMode)

    def redrawAll(self, canvas):
        numButtons = len(self.buttons)
        height = 25
        width = 50
        margin = 5
        y = self.app.height/2
        buttonSpace = numButtons*50+(numButtons-1)*margin
        for i in range(len(self.buttons)):
            x = (self.app.width-buttonSpace)/2+i*(width+margin)
            self.buttons[i].draw(canvas,x,y,height,width)
            self.buttons[i].xs = [x,x+width]
            self.buttons[i].ys = [y,y+height]
        canvas.create_text(self.app.width/2,self.app.height/2-margin,anchor="s",text=self.displayString)
        canvas.create_text(self.app.width/2,self.app.height/4,text="Press d when done!")

class PromptMode(Mode):
    def appStarted(self):
        pass
    def keyPressed(self, event):
        self.app.layers = int(self.getUserInput('How many hidden layers?'))
        self.app.hiddenSizes = []
        for i in range(self.app.layers):
            self.app.hiddenSizes.append(int(self.getUserInput(f'How many neurons in hidden layer {i+1}?')))
        self.app.learnRate = float(self.getUserInput('Learning rate?'))
        self.app.interval = float(self.getUserInput('Approximation Interval?'))
        self.app.testSplit = float(self.getUserInput('%Testing?'))
        self.app.valSplit = float(self.getUserInput('%Validation?'))
        self.app.setActiveMode(self.app.functionMode)
    def redrawAll(self,canvas):
        canvas.create_text(self.app.width/2,self.app.height/2,text="Press any key to begin!")

def runPlayground():
    Playground(width=600,height=600)

def main():
    runPlayground()

if __name__ == '__main__':
    main()