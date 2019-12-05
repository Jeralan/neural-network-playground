from cmu_112_graphics import *
#from http://www.cs.cmu.edu/~112/notes/notes-animations-part1.html
from tkinter import *
import numpy as np
import os

#Neural Network Playground Modal App

class Playground(ModalApp):
    def appStarted(self):
        self.promptMode =  PromptMode()
        self.gameMode = GameMode()
        self.functionMode = FunctionMode()
        self.finalMode = FinalMode()
        self.inputMode = InputMode()
        self.setActiveMode(self.promptMode)
        self.timerDelay = 1

class DataButton(Mode):
    def __init__(self,x,y,name):
        self.x = x
        self.y = y
        self.name = name

    def draw(self,canvas):
        canvas.create_rectangle(self.x-40,self.y-10,self.x+40,self.y+10)
        canvas.create_text(self.x,self.y,text=self.name)

class InputMode(Mode):
    def appStarted(self):
        self.x = None
        self.dataFiles = []
        for npyFile in os.listdir():
            if npyFile[-4:] == ".npy":
                self.dataFiles.append(npyFile)
        fileSpace = len(self.dataFiles)*90
        minX = self.app.width/2-fileSpace/2
        minY = self.app.height/2
        self.dataButtons = []
        for i in range(len(self.dataFiles)):
            self.dataButtons.append(DataButton(minX+90*i+40,minY+10,self.dataFiles[i]))
    
    def mousePressed(self,event):
        for button in self.dataButtons:           
            if abs(event.x-button.x) <= 40 and abs(event.y-button.y) <= 10:
                if self.x == None:
                    self.x = list(np.load(button.name))
                    for i in range(len(self.x)):
                        self.x[i] = list(self.x[i])
                else:
                    #custom inputted data gets split up into testing, validation, and training in order
                    xList = self.x
                    yList = list(np.load(button.name))
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

    def redrawAll(self,canvas):
        fileSpace = len(self.dataFiles)*90
        minX = self.app.width/2-fileSpace/2
        minY = self.app.height/2
        if self.x == None:
            canvas.create_text(self.app.width/2,minY/2,text="Select .npy file for input:",font="arial 60 bold")
        else:
            canvas.create_text(self.app.width/2,minY/2,text="Select .npy file for output:",font="arial 60 bold")
        for button in self.dataButtons:
            button.draw(canvas)

class FinalMode(Mode):
    def redrawAll(self,canvas):
        canvas.create_text(self.app.width/2,self.app.height/2,text=f"{round(self.app.yTestError,5)}",font="arial 60 bold", anchor="n")
        canvas.create_text(self.app.width/2,self.app.height/2-20,text="Test Mean Squared Error:",font="arial 40", anchor="s")
        canvas.create_text(self.app.width/2,self.app.height-20,text="Press any key to restart!",font="arial 20",anchor="s")

    def keyPressed(self,event):
        self.app.appStarted()

def tupleReLU(L):
    L = list(L)
    for x in range(len(L)):
        if L[x] < 0: L[x] = 0
    return tuple(L)

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
        if isinstance(self.app.xTrain[0],int):
            #normalizing data if it has only one input neuron
            self.customInput = False
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
        else: 
            self.customInput = True
        self.layers = len(self.hiddenSizes)
        self.timerDelay = 1
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
                    neuronWeights = []
                    if self.customInput:
                        for neuron in range(len(self.xTrain[0])):
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
                    if self.customInput:
                        yTrainPred = [self.forward(x,self.weights,self.biases)[0] for x in self.xTrain]
                    else:
                        yTrainPred = [self.forward([x],self.weights,self.biases)[0] for x in self.xTrain]
                    yTrainError = meanSquared(yTrainPred,self.yTrain)
                    newWeights[i][j][k] += (-(yTrainError-self.yTrainError)/self.app.interval)*self.app.learnRate
                    self.weights[i][j][k] -= self.app.interval
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                self.biases[i][j] += self.app.interval
                if self.customInput:
                    yTrainPred = [self.forward(x,self.weights,self.biases)[0] for x in self.xTrain]
                else:
                    yTrainPred = [self.forward([x],self.weights,self.biases)[0] for x in self.xTrain]
                yTrainError = meanSquared(yTrainPred,self.yTrain)
                newBiases[i][j] += (-(yTrainError-self.yTrainError)/self.app.interval)*self.app.learnRate
                self.biases[i][j] -= self.app.interval
        return (newWeights,newBiases)
    
    def forward(self,x,weights,biases):
        return self.reForward(tupleize(x),tupleize(weights),tupleize(biases),0)

    def reForward(self,x,weights,biases,depth):
        if (x,weights,biases,depth) in self.cachedResults:
            return self.cachedResults[(x,weights,biases,depth)]
        if len(weights) == 0:
            self.cachedResults[(x,weights,biases,depth)] = x
            return x
        prevLayer = self.reForward(x,weights[:-1],biases[:-1],depth-1)
        layerOutput = []
        for i in range(len(weights[-1])):
            neuron = np.dot(prevLayer,weights[-1][i])
            neuron += biases[-1][i]
            layerOutput.append(neuron)
        if depth == 0: 
            self.cachedResults[(x,weights,biases,depth)] = layerOutput
            return layerOutput
        else: 
            self.cachedResults[(x,weights,biases,depth)] = tupleReLU(layerOutput)
            return tupleReLU(layerOutput)
        
    def timerFired(self):
        self.timer += 1
        self.cachedResults = dict()
        if self.customInput:
            yTrainPred = [self.forward(x,self.weights,self.biases)[0] for x in self.xTrain]
            yValPred = [self.forward(x,self.weights,self.biases)[0] for x in self.xVal]
        else:
            yTrainPred = [self.forward([x],self.weights,self.biases)[0] for x in self.xTrain]
            yValPred = [self.forward([x],self.weights,self.biases)[0] for x in self.xVal]
        self.yTrainPred = yTrainPred
        self.yValPred = yValPred
        self.yTrainError = meanSquared(yTrainPred,self.yTrain)
        self.yValError = meanSquared(yValPred,self.yVal)
        self.valErrors.append(self.yValError)
        self.trainErrors.append(self.yTrainError)
        self.weights,self.biases = self.backward()

    def drawNeurons(self, canvas):
        layers = len(self.weights)+1
        spacing = self.networkArea/(2*layers+1)
        r = self.r
        cx,cy = spacing+r-spacing+r/2,self.app.height/2
        if self.customInput:
            height = len(self.xTrain[0])*(spacing+r*2)
            minY = self.app.height/2-height/2
            maxY = self.app.height/2+height/2
            for i in range(len(self.xTrain[0])):
                cy = self.app.height/(len(self.xTrain[0])+1)*(i+1)
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="black")
        else:
            canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="black")
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                cx,cy = spacing*3+spacing*2*i+r-spacing+r/2,self.app.height/(len(self.biases[i])+1)*(j+1)
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="black")
                if self.biases[i][j] >= 0:
                    text = f"+{round(self.biases[i][j],3)}"
                else:
                    text = f"{round(self.biases[i][j],3)}"
                canvas.create_text(cx,cy,text=text,fill="white",anchor="c",font=f"arial {int(r/2)}")

    def drawWeights(self, canvas):
        if self.customInput:
            layerSizes = [len(self.xTrain[0])]+self.hiddenSizes
        else:
            layerSizes = [1] + self.hiddenSizes
        layers = len(self.weights)+1
        spacing = self.networkArea/(2*layers+1)
        r = (self.app.height/(max(max(self.hiddenSizes),len(self.hiddenSizes))*2+2)*2/3)
        self.r = r
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    width = abs(self.weights[i][j][k])*r
                    if self.weights[i][j][k] > 0: color = "red"
                    else: color = "blue"
                    x1,y1 = spacing*3+spacing*2*(i-1)+r-spacing+r/2,self.app.height/(layerSizes[i]+1)*(k+1)
                    x2,y2 = spacing*3+spacing*2*i+r-spacing+r/2,self.app.height/(layerSizes[i+1]+1)*(j+1)
                    canvas.create_line(x1,y1,x2,y2,width=width,fill=color)
    
    def drawError(self, canvas):
        errorWidth = self.app.width/3
        errorHeight = self.app.height/2
        points = min(len(self.valErrors),int(errorWidth))
        if points > 1:
            maxError = max(max(self.valErrors),max(self.trainErrors))
            r = 2
            maxHeight = 0
            for i in range(1,points+1):
                cx = self.app.width - errorWidth*(i/points)
                cy = errorHeight - (self.valErrors[-i]/maxError)*errorHeight + 10
                maxHeight = max(cy,maxHeight)
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="red",outline="red")
                cy = errorHeight - (self.trainErrors[-i]/maxError)*errorHeight + 10
                maxHeight = max(cy,maxHeight)
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="black")
            cx = self.app.width-errorWidth
            cy = maxHeight
            canvas.create_text(cx-20,cy,anchor="ne",text=f"{min(min(self.valErrors[-points:]),min(self.trainErrors[-points:]))}")
            cy = 0
            canvas.create_text(cx-20,cy,anchor="ne",text=f"{maxError}")
            cy = errorHeight + 10
            canvas.create_line(cx,cy,cx+self.app.width,cy,fill="black")
            canvas.create_text(cx,cy,anchor="ne",text="0")
            cx = self.app.width-5
            cy = 5
            canvas.create_text(cx,cy,text="Training",font="bold 10",anchor="ne")
            canvas.create_text(cx,cy+25,text="Validation",font="bold 10",anchor="ne",fill="red")
            cx = self.app.width-errorWidth/2
            canvas.create_text(cx,cy,text="Error",font="bold 20",anchor="n")

    def drawFunction(self, canvas):
        if self.timer > 2:
            minWidth = self.app.width*2/3
            maxWidth = self.app.width-10
            inputs = len(self.xTrain) + len(self.xVal)
            centerWidth = (maxWidth+minWidth)/2
            maxWidth = min(maxWidth,inputs/2*30+centerWidth)
            minWidth = max(minWidth,centerWidth-inputs/2*30)
            minHeight = self.app.height/2+40
            maxHeight = self.app.height
            centerHeight = (maxHeight+minHeight)/2
            maxHeight = inputs*3+centerHeight
            minHeight = centerHeight-inputs*3
            minX = min(min(self.xTrain),min(self.xVal))
            maxX = max(max(self.xTrain),max(self.xVal))
            maxY = max(max(self.yTrain),max(self.yVal))
            r = 100/inputs
            x = self.xTrain+self.xVal
            for i in range(len(self.xTrain)):
                cx = (self.xTrain[i]-minX)*((maxWidth-minWidth)/(maxX-minX))+minWidth
                cy = -self.yTrain[i]/maxY*(maxHeight-minHeight)+maxHeight
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="black")
                cy = -self.yTrainPred[i]/maxY*(maxHeight-minHeight)+maxHeight
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="green",outline="green")
            for i in range(len(self.xVal)):
                cx = (self.xVal[i]-minX)*((maxWidth-minWidth)/(maxX-minX))+minWidth
                cy = -self.yVal[i]/maxY*(maxHeight-minHeight)+maxHeight
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="black")
                cy = -self.yValPred[i]/maxY*(maxHeight-minHeight)+maxHeight
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill="green",outline="green")
            canvas.create_text(centerWidth,self.app.height,anchor="se",fill="green",text="Prediction ",font="arial 10 bold")
            canvas.create_text(centerWidth,self.app.height,anchor="sw",fill="black",text=" Actual",font="arial 10 bold")

    def redrawAll(self, canvas):
        self.networkArea = self.app.width*1/2
        self.drawWeights(canvas)
        self.drawNeurons(canvas)
        self.drawError(canvas)
        if not self.customInput:
            self.drawFunction(canvas)
        canvas.create_rectangle(0,0,50,20,fill="white")
        canvas.create_text(25,10,text="Done")
        canvas.create_rectangle(60,0,160,20,fill="white")
        canvas.create_text(110,10,text="Change Learn Rate")
        canvas.create_rectangle(170,0,270,20,fill="white")
        canvas.create_text(220,10,text="Add Neuron")
        canvas.create_rectangle(280,0,380,20,fill="white")
        canvas.create_text(330,10,text="Remove Neuron")
        canvas.create_rectangle(390,0,490,20,fill="white")
        canvas.create_text(440,10,text="Add Layer")
        canvas.create_rectangle(500,0,600,20,fil="white")
        canvas.create_text(550,10,text="Remove Layer")
    
    def mousePressed(self, event):
        if event.x <= 50 and event.y <= 20:
            if self.customInput:
                yTestPred = [self.forward(x,self.weights,self.biases)[0] for x in self.xTest]
            else:
                yTestPred = [self.forward([x],self.weights,self.biases)[0] for x in self.xTest]
            self.app.yTestError = meanSquared(yTestPred,self.yTest)
            self.app.setActiveMode(self.app.finalMode)
        elif event.x >= 60 and event.x <= 160 and event.y <= 20:
            newRate = prompt(self,f'Current Learn Rate = {self.app.learnRate} (Float less than 1)',float," <= 1")
            if newRate != None: self.app.learnRate = newRate
        elif event.x >= 170 and event.x <= 270 and event.y <= 20:
            if len(self.weights)-1 >= 1:
                layer = 0
                while layer != None and layer <= 1:
                    layer = prompt(self,f'Which layer do you want to add a neuron too? (2-{len(self.weights)})',int,"<=len(self.weights)")
                if layer != None:
                    index = layer-2
                    if layer == 2:
                        inputNeurons = len(self.weights[0][0])
                    else:
                        inputNeurons = len(self.weights[index-1])
                    outputNeurons = len(self.weights[index+1])
                    inputWeights = []
                    for i in range(inputNeurons):
                        inputWeights.append(random.randint(-100,100)/100)
                    self.weights[index].append(inputWeights)
                    for i in range(outputNeurons):
                        self.weights[index+1][i].append(random.randint(-100,100)/100)
                    self.biases[index].append(0)
                    self.hiddenSizes[index] += 1
        elif event.x >= 280 and event.x <= 380 and event.y <= 20:
            if len(self.weights)-1 >= 1:
                layer = 0
                while layer != None and layer <= 1:
                    layer = prompt(self,f'Which layer do you want to remove a neuron from? (2-{len(self.weights)})',int,"<=len(self.weights)")
                if layer != None:
                    index = layer-2
                    if self.hiddenSizes[index] > 1:
                        if layer == 2:
                            inputNeurons = len(self.weights[0][0])
                        else:
                            inputNeurons = len(self.weights[index-1])
                        outputNeurons = len(self.weights[index+1])
                        self.weights[index].pop(-1)
                        for  i in range(outputNeurons):
                            self.weights[index+1][i].pop(-1)
                        self.biases[index].pop(-1)
                        self.hiddenSizes[index] -= 1
        elif event.x >= 390 and event.x <= 490 and event.y <= 20:
            self.weights.append([[random.randint(-100,100)/100]])
            self.biases.append([0])
            self.hiddenSizes.append(1)
        elif event.x >= 500 and event.x <= 600 and event.y <= 20:
            if len(self.weights)-1 >= 1:
                layer = 0
                while layer != None and layer <= 1:
                    layer = prompt(self,f'Which layer do you want to remove? (2-{len(self.weights)})',int,"<=len(self.weights)")
                if layer != None:
                    index = layer-2
                    if layer == 2:
                        inputNeurons = len(self.weights[0][0])
                    else:
                        inputNeurons = len(self.weights[index-1])
                    outputNeurons = len(self.weights[index+1])
                    newWeights = []
                    for i in range(outputNeurons):
                        outputNeuron = []
                        for j in range(inputNeurons):
                            outputNeuron.append(random.randint(-100,100)/100)
                        newWeights.append(outputNeuron)
                    self.weights.pop(index)
                    self.weights[index] = newWeights
                    self.biases.pop(index) 
                    self.hiddenSizes.pop(index)


import random
def createXList(left,right,num):
    xList = random.sample(range(left,right),num)
    return xList

class FunctionButton(object):
    def __init__(self,function,name):
        self.function = function
        self.name = name
    def applyFunction(self,functionString,displayString):
        return (functionString + self.function,displayString + self.name)
    def draw(self,canvas,x,y,h,w):
        canvas.create_rectangle(x,y,x+w,y+h,fill="white",width=2)
        canvas.create_text(x+w/2,y+h/2,text=self.name)

def prompt(self, message, target, constraint):
    output = "None"
    while not isinstance(output, target):
        output = self.getUserInput(message)
        if output == None:
            return output
        elif target == float:
            if output.replace('.','',1).isdigit():
                if eval(output+constraint):
                    output = float(output)
        elif target == int: 
            if output.replace('-',"",1).isdigit():
                if eval(output+constraint):
                    output = int(output)
    return output

import math
class FunctionMode(Mode):
    def appStarted(self):
        self.timerDelay = 1
        self.app.xList = None
        self.functionString = ""
        self.displayString = ""
        self.buttons = [FunctionButton("math.cos","Cos"),
                        FunctionButton("math.sin","Sin"),
                        FunctionButton("math.tan","Tan"),
                        FunctionButton("math.exp","Exp"),
                        FunctionButton("x","x"),
                        FunctionButton("*","*"),
                        FunctionButton("/","/"),
                        FunctionButton("+","+"),
                        FunctionButton("-","-"),
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
                        FunctionButton("(","("),
                        FunctionButton(")",")"),]
    def mousePressed(self,event):
        if event.x <= 80 and event.y <= 20:
            self.app.setActiveMode(self.app.inputMode)
        elif self.app.xList == None:
            self.Left,self.Right = 0,0
            while self.Left == None or self.Right == None or int((self.Right-self.Left)*self.app.testSplit) < 1:
                self.Left = prompt(self,'Integer Left Bound of X?', int, " != 0.1")
                self.Right = prompt(self,'Integer Right Bound of X?',int,f'> {self.Left}')
            self.num = self.Right-self.Left
            self.app.xList = createXList(self.Left,self.Right,self.num)
            self.app.yList = self.app.xList
        for i in range(len(self.buttons)):
            if event.x > self.buttons[i].xs[0] and event.x < self.buttons[i].xs[1]:
                if event.y > self.buttons[i].ys[0] and event.y < self.buttons[i].ys[1]:
                    self.functionString,self.displayString = self.buttons[i].applyFunction(self.functionString,self.displayString)
    def keyPressed(self,event):
        if event.key == "d":
            fail = False
            yList = []
            for element in self.app.xList:
                x = element
                try:
                    yList.append(float(eval(self.functionString)))
                except:
                    fail = True
                    break
            if not fail:
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
            else:
                self.appStarted()

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
        canvas.create_rectangle(0,0,80,20,fill="white")
        canvas.create_text(40,10,text="Input Data")

class PromptMode(Mode):
    def appStarted(self):
        pass
    def keyPressed(self, event):
        self.app.layers = None
        while self.app.layers == None:
            self.app.layers = prompt(self,'How many hidden layers? (Non-Negative Integer)',int," >= 0")
        self.app.hiddenSizes = []
        for i in range(self.app.layers):
            newSize = None
            while newSize == None:
                newSize = prompt(self,f'How many neurons in hidden layer {i+1}? (Positive Integer)',int," > 0")
            self.app.hiddenSizes.append(newSize)
        self.app.learnRate = None
        while self.app.learnRate == None:
            self.app.learnRate = prompt(self,'Learning rate? (Float less than 1)',float," <= 1")
        self.app.interval = 1e-15
        self.app.testSplit = 0.2
        self.app.valSplit = None
        while self.app.valSplit == None or self.app.valSplit > 50:
            self.app.valSplit = prompt(self,'Percentage of Data used for Validation? (1-50)',int," >= 1")
        self.app.valSplit /= 100
        self.app.setActiveMode(self.app.functionMode)
    def redrawAll(self,canvas):
        canvas.create_rectangle(0,0,self.app.width,self.app.height,fill="purple")
        canvas.create_text(self.app.width/2,self.app.height/4,anchor="n",text="Neural Network Playground",font=f"arial {self.app.width//20} bold")
        canvas.create_text(self.app.width/2,self.app.height/2,text="Press any key to begin!",font="arial 20")

def runPlayground():
    Playground(width=600,height=600)

def main():
    runPlayground()

if __name__ == '__main__':
    main()