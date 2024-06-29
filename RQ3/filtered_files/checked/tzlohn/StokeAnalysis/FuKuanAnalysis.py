import yfinance as yf
import numpy as np
from PyQt5 import QtCore,QtWidgets,QtGui
from  matplotlib import pyplot as plt
import sys,csv
import CrossHairCursor as CHC
import operator
from matplotlib import pyplot

OPDict = {">":operator.gt,"=":operator.eq,">":operator.lt}

class DataSourceGroup(QtWidgets.QGroupBox):
    def __init__(self,parent):
        super().__init__()

        self.MainWin = parent
        self.setTitle("資料設定")

        self.TickerLabel = QtWidgets.QLabel(parent = self,text = "代碼:")
        self.TickerBox = QtWidgets.QComboBox(self)
        self.TickerBox.addItems(["0050.TW","2330.TW","^TWII","EURUSD=X","USDTWD=X","EURTWD=X"])
        self.TickerBox.currentIndexChanged.connect(self.getTicket)
        self.PeriodLabel = QtWidgets.QLabel(parent = self,text = "載入資料期間(從現在回朔):")
        self.DigitBox = QtWidgets.QSpinBox(self)
        self.DigitBox.setValue(1)
        self.DigitBox.setMinimum(1)
        self.DigitBox.setMaximum(12)
        self.DigitBox.valueChanged.connect(self.resetPeriod)
        self.UnitBox = QtWidgets.QComboBox(self)
        self.UnitBox.addItems(["日","月","年"])
        self.UnitBox.setCurrentIndex(1)
        self.UnitBox.currentIndexChanged.connect(self.changeDigitList)
        self.UnitBox.currentIndexChanged.connect(self.resetPeriod)
        self.getTicket()

        self.StartDateLabel = QtWidgets.QLabel(parent = self, text = "開始日期 (yyyy-mm-dd):")
        self.StartDate = QtWidgets.QLineEdit(self)
        self.StartDate.setText(str(self.Data.index.date[0]))
        self.EndDateLabel = QtWidgets.QLabel(parent = self, text = "結束日期 (yyyy-mm-dd):")
        self.EndDate = QtWidgets.QLineEdit(self)
        self.EndDate.setText(str(self.Data.index.date[-1]))

        self.StartButton = QtWidgets.QPushButton("更新資料")
        self.StartButton.clicked.connect(self.getData)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.addWidget(self.TickerLabel,0,0,2,2)
        self.layout.addWidget(self.TickerBox,0,2,2,2)
        self.layout.addWidget(self.PeriodLabel,2,0,2,2)
        self.layout.addWidget(self.DigitBox,2,2,2,1)
        self.layout.addWidget(self.UnitBox,2,3,2,1)
        self.layout.addWidget(self.StartDateLabel,4,0,2,2)
        self.layout.addWidget(self.StartDate,4,2,2,2)
        self.layout.addWidget(self.EndDateLabel,6,0,2,2)
        self.layout.addWidget(self.EndDate,6,2,2,2)
        self.layout.addWidget(self.StartButton,8,1,2,2)

        self.setLayout(self.layout)
    
    def changeDigitList(self):
        if self.UnitBox.currentIndex() == 0:
            self.DigitBox.setValue(1)
            self.DigitBox.setMinimum(1)
            self.DigitBox.setMaximum(31)
        if self.UnitBox.currentIndex() == 2:
            self.DigitBox.setValue(1)
            self.DigitBox.setMinimum(1)
            self.DigitBox.setMaximum(20)
        if self.UnitBox.currentIndex() == 1:
            self.DigitBox.setValue(1)
            self.DigitBox.setMinimum(1)
            self.DigitBox.setMaximum(12)

    def transferUnit(self):
        if self.UnitBox.currentIndex() == 0:
            Unit = "d"
        if self.UnitBox.currentIndex() == 2:
            Unit = "y"
        if self.UnitBox.currentIndex() == 1:
            Unit = "mo"
        return Unit

    def getTicket(self):
        self.Ticket = yf.Ticker(self.TickerBox.currentText())
        self.Unit = self.transferUnit()
        self.Period = str(self.DigitBox.value()) + self.Unit
        self.Data = self.Ticket.history(period = self.Period)
        #print(self.Data)
    
    def resetPeriod(self):
        self.Unit = self.transferUnit()
        self.Period = str(self.DigitBox.value()) + self.Unit
        self.Data = self.Ticket.history(period = self.Period)
        #完整資料history(interval = "1m")
        #data interval (1m data is only for available for last 7 days, and data interval <1d for the last 60 days) Valid intervals are:
        #“1m”, “2m”, “5m”, “15m”, “30m”, “60m”, “90m”, “1h”, “1d”, “5d”, “1wk”, “1mo”, “3mo”
        self.resetDate()

    def resetDate(self):
        self.StartDate.setText(str(self.Data.index.date[0]))
        self.EndDate.setText(str(self.Data.index.date[-1]))

    def updateTicket(self):
        self.PeriodData = self.Data.loc[self.StartDate.text():self.EndDate.text()]
        self.Date = self.PeriodData.axes[0].to_list()

    def getData(self):
        self.updateTicket()
        #self.DefaultAnalysis() 
        self.MainWin.Metric.setEnabled(True)
        self.MainWin.Condition.setEnabled(True)  
        self.MainWin.Metric.packData(self.PeriodData)     
    
    def DefaultAnalysis(self):
        High = self.PeriodData["High"]
        Low = self.PeriodData["Low"]
        Close = self.PeriodData["Close"]

        Cash = 1
        BuyCount = 0
        SellCount = 0
        for n in range(1,len(High)):
            if Close[n] > High[n-1]:
                Cash = Cash - 1 # Buy
                BuyCount = BuyCount + 1
            elif Close[n] < Low[n-1]:
                Cash = Cash + 1 # Sell
                SellCount = SellCount +1
        
        Message = "這段期間裡你賺了 %d元, 其中包含了 %d次 的買進 和 %d次的 賣出"%(Cash, BuyCount, SellCount)
        QtWidgets.QMessageBox.information(None,"Result",Message)

class ColorDelegate(QtWidgets.QStyledItemDelegate):
    """This class is generated by chatGPT"""
    def paint(self, painter, option, index):
        # Get the color data from the model
        color = index.data(QtCore.Qt.UserRole)

        # Draw the color rectangle
        rect = option.rect.adjusted(2, 2, -20, -2)
        painter.fillRect(rect, QtGui.QColor(color))

        # Draw the item text
        text_rect = QtCore.QRect(rect.right() + 5, rect.top(), 200, rect.height())
        painter.drawText(text_rect, QtCore.Qt.AlignVCenter, index.data(QtCore.Qt.DisplayRole))


def convertQColorToQIcon(color):
    # color: QColor 
    # Create a QPixmap object with the desired size and color
    sizeV = 15  # desired size of the icon in pixels
    sizeH = 20
    pixmap = QtGui.QPixmap(sizeH, sizeV)
    pixmap.fill(color)

    # Create a QIcon object from the pixmap
    icon = QtGui.QIcon(pixmap)

    return icon

def ColorSelectComboBox(self,ColorList):
    ColorBox = QtWidgets.QComboBox(self)
    for color in ColorList:
        ColorBox.addItem(convertQColorToQIcon(QtGui.QColor(color)),"")    
    return ColorBox        

class MetricGroup(QtWidgets.QGroupBox):
    def __init__(self,parent):
        super().__init__()

        self.MainWin = parent
        self.setTitle("技術線型")
        self.OptionDict ={0:"Open",1:"Close",2:"High",3:"Low"}
           
        self.MainWin.sig_FoundPoints.connect(self.plotBlocks)

        self.RawCB = QtWidgets.QCheckBox(self)
        self.RawCB.setText("完整資料")

        self.DayBoxCB = QtWidgets.QCheckBox(self)
        self.DayBoxCB.setText("日K線")

        self.DayLineCB = QtWidgets.QCheckBox(self)
        self.DayLineCB.setText("日線")

        self.DayLineOpt = QtWidgets.QComboBox(self)
        self.DayLineOpt.addItems(["開盤","收盤","最高","最低","高低平均","收盤-10日"])

        self.DayLineColor = ColorSelectComboBox(self,self.MainWin.ColorList)               

        self.AvgCB = QtWidgets.QCheckBox(self)
        self.AvgCB.setText("平均線")
        self.AvgCB.clicked.connect(self.checkAvgBox)

        self.AvgDaysBox = QtWidgets.QSpinBox(self)
        self.AvgDaysBox.setMinimum(3)
        self.AvgDaysBox.setMaximum(20)
        self.AvgDaysBox.setValue(10)
        self.AvgDaysBox.setDisabled(True)

        self.AvgDaysText = QtWidgets.QLabel(parent = self)
        self.AvgDaysText.setText("天平均")

        self.AvgDaysColor = ColorSelectComboBox(self,self.MainWin.ColorList)
        
        self.AvgMonCB = QtWidgets.QCheckBox(self)
        self.AvgMonCB.setText("月平均線")
        self.AvgMonCB.setDisabled(True)

        self.AvgMonColor = ColorSelectComboBox(self,self.MainWin.ColorList)

        self.RSICB = QtWidgets.QCheckBox(self)
        self.RSICB.setText("RSI")
        self.RSICB.clicked.connect(self.checkRSIBox)

        self.RSIDaysBox = QtWidgets.QSpinBox(self)
        self.RSIDaysBox.setMinimum(3)
        self.RSIDaysBox.setMaximum(20)
        self.RSIDaysBox.setValue(10)
        self.RSIDaysBox.setDisabled(True)

        self.RSIDaysText = QtWidgets.QLabel(parent = self)
        self.RSIDaysText.setText("天RSI")
        
        self.RSIDaysColor = ColorSelectComboBox(self,self.MainWin.ColorList)

        self.PlotButton = QtWidgets.QPushButton(self)
        self.PlotButton.setText("觀察技術線")
        self.PlotButton.clicked.connect(self.plotCurves)

        self.Layout = QtWidgets.QGridLayout(self)
        self.Layout.addWidget(self.RawCB,0,0,1,2)
        self.Layout.addWidget(self.DayBoxCB,0,2,1,2)
        self.Layout.addWidget(self.DayLineCB,1,0,1,2)
        self.Layout.addWidget(self.DayLineOpt,1,2,1,1)
        self.Layout.addWidget(self.DayLineColor,1,4,1,1)
        self.Layout.addWidget(self.AvgCB,2,0,1,2)
        self.Layout.addWidget(self.AvgDaysBox,2,2,1,1)
        self.Layout.addWidget(self.AvgDaysText,2,3,1,1)
        self.Layout.addWidget(self.AvgDaysColor,2,4,1,1)
        self.Layout.addWidget(self.AvgMonCB,3,0,1,2)
        self.Layout.addWidget(self.AvgMonColor,3,4,1,1)
        self.Layout.addWidget(self.RSICB,4,0,1,2)
        self.Layout.addWidget(self.RSIDaysBox,4,2,1,1)
        self.Layout.addWidget(self.RSIDaysText,4,3,1,1)
        self.Layout.addWidget(self.RSIDaysColor,4,4,1,1)
        self.Layout.addWidget(self.PlotButton,5,1,1,2)

        self.setLayout(self.Layout)
        self.CrossHairPlot = CHC.CrosshairPlotWidget(parent = self, title = "本和里發財燒臘")
        self.isBoxChartPlotted = False

    def plotCurves(self):
        self.CrossHairPlot.PlotWidget.show()

        if self.DayBoxCB.isChecked() and not self.isBoxChartPlotted:
            self.CrossHairPlot.plotBoxChart(self.PackedRaw,self.Date)
            self.isBoxChartPlotted = True
        
        if hasattr(self.CrossHairPlot,"ThisPlot"):
            for item in self.CrossHairPlot.ThisPlot:
                self.CrossHairPlot.PlotWidget.plotItem.removeItem(item)            

        if self.DayLineCB.isChecked():
            Option = self.DayLineOpt.currentIndex()
            QColor = self.MainWin.ColorList[self.DayLineColor.currentIndex()]
            match Option:
                case 4:
                    self.RawData = (self.MainWin.DataSource.PeriodData["High"].values + self.MainWin.DataSource.PeriodData["Low"].values)/2
                    self.CrossHairPlot.plot(self.RawData,np.arange(self.RawData.shape[0]),QColor)
                case 5:
                    RawData = self.MainWin.DataSource.PeriodData["Close"]
                    self.calculateMetric(type = "DayAvg",Data = RawData)
                    Data = RawData[9:]-self.RollingAvg[0]
                    self.CrossHairPlot.plot(Data,self.RollingAvg[1],QColor,isRight = True)
                case _:
                    self.RawData = self.MainWin.DataSource.PeriodData[self.OptionDict[Option]]
                    self.CrossHairPlot.plot(self.RawData.values,np.arange(self.RawData.values.shape[0]),QColor)


        if self.AvgCB.isChecked():
            if self.DayLineCB.isChecked() and self.DayLineOpt.currentIndex() == 4:
                RawData = self.RawData
                self.calculateMetric(type = "DayAvg",Data = RawData)
            else:
                RawData = self.MainWin.DataSource.PeriodData["Close"]
                self.calculateMetric(type = "DayAvg",Data = RawData.values)
            QColor = self.MainWin.ColorList[self.AvgDaysColor.currentIndex()]
            #self.CrossHairPlot.PlotWidget.plotItem.showAxis("left")
            self.CrossHairPlot.plot(self.RollingAvg[0],self.RollingAvg[1],QColor)
        
    def packData(self,RawData):
        OpenData = RawData["Open"]
        CloseData = RawData["Close"]
        HighData = RawData["High"]
        LowData = RawData["Low"]

        self.Date = self.getDate(LowData)
        self.PackedRaw = [(O,C,H,L) for O,C,H,L in zip(OpenData,CloseData,HighData,LowData)]

    def getDate(self,TicketData):
        Date = TicketData.axes[0]
        return [str(date)[0:10] for date in Date]

    def calculateMetric(self,type,Data):
        if type  == "DayAvg":
            AvgDays = self.AvgDaysBox.value()
            Kernel = self.createKernel(type = "DayAvg",AvgDay = AvgDays,Length = len(Data))
            RollingAvg = np.dot(Data,Kernel)/AvgDays
            #self.RollingAvg = np.concatenate((np.asarray([0]*AvgDays),self.RollingAvg))
            self.RollingAvg = [RollingAvg,np.arange(AvgDays-1,len(Data),1)]

    def createKernel(self,type,AvgDay,Length):
        if type == "DayAvg":  
            Kernel = np.zeros(shape = (Length,Length-AvgDay+1))
            for n in range(Length-AvgDay+1):
                Kernel[n:n+AvgDay,n] = 1
            return Kernel

    def checkAvgBox(self):
        if self.AvgDaysBox.isEnabled():
            self.AvgDaysBox.setDisabled(True)
        else:
            self.AvgDaysBox.setEnabled(True)
    
    def checkRSIBox(self):
        if self.RSIDaysBox.isEnabled():
            self.RSIDaysBox.setDisabled(True)
        else:
            self.RSIDaysBox.setEnabled(True)
    
    QtCore.pyqtSlot(list)
    def plotBlocks(self,FoundPoints):
        if hasattr(self.CrossHairPlot,"Lines"):
            for item in self.CrossHairPlot.Lines:
                self.CrossHairPlot.PlotWidget.plotItem.removeItem(item)

        for points in FoundPoints:
            match points[2]:
                case "In":
                    LineThick = 0.05
                case "Out":
                    LineThick = 0.06
            
            self.CrossHairPlot.plotZone(points[0],0,points[1],LineThick)
            #for point in points[0]:
            #    print(point,self.Date[int(round(point))])

class Condition(QtWidgets.QGroupBox):
    def __init__(self,parent,name = None):
        super().__init__()
        self.ConditionWin = parent
        self.setTitle(name)
        self.RowNo = 0

        self.MetricsOption1 = list()
        self.MetricsOption2 = list()
        self.MetricsCondition1 = list()
        self.MetricsCondition2 = list()
        self.DayOffset1 = list()
        self.DayOffset2 = list()
        self.Operator = list()
        self.AndOr = list()
        self.LabelColor = list()

        self.addWidgets()

    def addWidgets(self):
        MetricsOption1 = QtWidgets.QComboBox(self)
        MetricsOption1.addItems(self.ConditionWin.MainWin.MetricList)
        MetricsOption1.currentTextChanged.connect(lambda func:self.changeMetricCondition(1,MetricsOption1.currentText(),self.RowNo))
        self.MetricsOption1.append(MetricsOption1)

        MetricsCondition1 = QtWidgets.QComboBox(self)
        MetricsCondition1.addItems(["開盤","收盤","最高","最低"])
        self.MetricsCondition1.append(MetricsCondition1)

        DayOffset1 = QtWidgets.QSpinBox(self)
        DayOffset1.setMinimum(-10)
        DayOffset1.setMaximum(0)
        self.DayOffset1.append(DayOffset1)

        DayOffset2 = QtWidgets.QSpinBox(self)
        DayOffset2.setMinimum(-10)
        DayOffset2.setMaximum(0)
        self.DayOffset2.append(DayOffset2)

        Condition = QtWidgets.QComboBox(self)
        Condition.addItems(["<","=",">"])
        self.Operator.append(Condition)

        MetricsOption2 = QtWidgets.QComboBox(self)
        MetricsOption2.addItems(self.ConditionWin.MainWin.MetricList)
        MetricsOption2.currentTextChanged.connect(lambda func:self.changeMetricCondition(2,MetricsOption2.currentText(),self.RowNo))
        self.MetricsOption2.append(MetricsOption2)

        MetricsCondition2 = QtWidgets.QComboBox(self)
        MetricsCondition2.addItems(["開盤","收盤","最高","最低"])
        self.MetricsCondition2.append(MetricsCondition2)

        AddConditionButton = QtWidgets.QPushButton(self)
        AddConditionButton.setText("增加條件")
        AddConditionButton.clicked.connect(self.addMoreWidget)

        AndOr = QtWidgets.QComboBox(self)
        AndOr.addItems(["or","and"])
        AndOr.setDisabled(True)
        self.AndOr.append(AndOr)

        LabelColor = ColorSelectComboBox(self,self.ConditionWin.MainWin.ColorList)
        self.LabelColor.append(LabelColor)

        if not hasattr(self,"Layout"):
            self.Layout = QtWidgets.QGridLayout(self)
        self.Layout.addWidget(MetricsOption1,self.RowNo,0,1,1)
        self.Layout.addWidget(DayOffset1,self.RowNo,1,1,1)
        self.Layout.addWidget(MetricsCondition1,self.RowNo,2,1,1)
        self.Layout.addWidget(Condition,self.RowNo,3,1,1)        
        self.Layout.addWidget(MetricsOption2,self.RowNo,4,1,1)
        self.Layout.addWidget(DayOffset2,self.RowNo,5,1,1)        
        self.Layout.addWidget(MetricsCondition2,self.RowNo,6,1,1)
        self.Layout.addWidget(LabelColor,self.RowNo,7,1,1)
        self.Layout.addWidget(AddConditionButton,self.RowNo+1,0,1,1)
        self.Layout.addWidget(AndOr,self.RowNo+1,1,1,1)
        
        self.setLayout(self.Layout)
    
    def addMoreWidget(self):
        self.RowNo = self.RowNo+2
        self.AndOr[-1].setEnabled(True)
        self.addWidgets() 

    def changeMetricCondition(self,idx,txt,ind):
        ind = int(ind/2)
        match idx:
            case 1:
                if txt == "日線":
                    self.MetricsCondition1[ind].setEnabled(True)
                    self.DayOffset1[ind].setEnabled(True)
                else:
                    self.MetricsCondition1[ind].setDisabled(True)
                    self.DayOffset1[ind].setEnabled(False)
            case 2:
                if txt == "日線":
                    self.MetricsCondition2[ind].setEnabled(True)
                    self.DayOffset2[ind].setEnabled(True)
                else:
                    self.MetricsCondition2[ind].setDisabled(True)
                    self.DayOffset2[ind].setEnabled(False)
            case _:
                pass

class ConditionGroup(QtWidgets.QGroupBox):
    def __init__(self,parent):
        super().__init__()
        self.MainWin = parent
        self.setTitle("條件分析")

        self.ConditionIn = Condition(self,"進場條件")
        self.ConditionOut = Condition(self,"出場條件")
        
        self.AnalyzeButton = QtWidgets.QPushButton(self)
        self.AnalyzeButton.setText("搜尋標的")
        self.AnalyzeButton.clicked.connect(self.getRange)

        self.LabelBox = QtWidgets.QCheckBox(self)
        self.LabelBox.setText("顯示於圖上")

        """
        self.PMLabel = QtWidgets.QLabel(parent = self, text = "標記+/-")
        self.DaysLabel = QtWidgets.QLabel(parent = self, text = "天")
        self.DaySpinBox = QtWidgets.QSpinBox(self)
        self.DaySpinBox.setValue(5)
        self.DaySpinBox.setMinimum(0)
        self.DaySpinBox.setMaximum(30)
        """
        self.TableOutput = QtWidgets.QPushButton(self)
        self.TableOutput.setText("生成報表")
        self.TableOutput.clicked.connect(self.outputTable)
        self.TableOutput.setDisabled(True)

        self.Layout = QtWidgets.QGridLayout(self)
        self.Layout.addWidget(self.ConditionIn,0,0,1,4)
        self.Layout.addWidget(self.ConditionOut,1,0,1,4)
        self.Layout.addWidget(self.TableOutput,2,0,1,1)
        self.Layout.addWidget(self.LabelBox,2,3,1,1)
        """
        self.Layout.addWidget(self.PMLabel,2,0,1,1)
        self.Layout.addWidget(self.DaySpinBox,2,1,1,1)
        self.Layout.addWidget(self.DaysLabel,2,2,1,1)
        """
        self.Layout.addWidget(self.AnalyzeButton,2,2,1,1)

        self.setLayout(self.Layout)
    
    def outputTable(self):

        PathDir = QtWidgets.QFileDialog.getExistingDirectory(self,"選擇資料夾")
        PathDir = PathDir + "/"+"報表.csv"
        
        #OpenData = self.MainWin.DataSource.PeriodData["Open"]
        #CloseData = self.MainWin.DataSource.PeriodData["Close"]
        HighData = self.MainWin.DataSource.PeriodData["High"]
        #LowData = self.MainWin.DataSource.PeriodData["Low"]
        Date = self.MainWin.Metric.getDate(HighData)
        BuyDict = self.dictPrice(self.BuyPrice)
        SellDict = self.dictPrice(self.SellPrice)
        TableList = list()
        
        for d in BuyDict:
            OutDict = dict()
            OutDict["日期"] = Date[d]
            OutDict["買入"] = round(BuyDict[d],2)
            OutDict["賣出"] = round(SellDict[d],2)
            OutDict["差價"] = round(SellDict[d]-BuyDict[d],2)
            TableList.append(OutDict)

        with open(PathDir,"w+") as csvFile:
            FieldNames = ["日期","買入","賣出","差價"]
            writer = csv.DictWriter(csvFile, fieldnames=FieldNames)

            writer.writeheader()
            for aDict in TableList:
                print(aDict)
                writer.writerow(aDict)    
            """
            for d,b,s,Diff in zip(DateList,BuyList,SellList,DiffList):
                writer.writerow({'日期': d, '買入': b, '賣出': s, '差價': Diff})
            """
        #for d,b,s,Diff in zip(DateList,BuyList,SellList,DiffList):
        #    print(d,b,s,Diff)
        #pyplot.plot(np.asarray(idxList),np.asarray(DiffList),"ro--")
        #pyplot.plot(np.asarray(idxList),np.zeros(len(idxList)))
        #pyplot.show()
    
    def dictPrice(self,PriceList):
        PriceDict = dict()
        UsedPoint = list()
        for APrice in PriceList:
            for aPoint in APrice:
                if aPoint[0] not in UsedPoint:
                    PriceDict[aPoint[0]] = aPoint[1]
                    UsedPoint.append(aPoint[0])
        
        return PriceDict
    
    def pickDataPoint(self,Obj,Status, ConditionPoints = None):
        Colors = list()
        FoundPoints = list()
        PriceList = list()

        Conditions = list()

        for idx in range(len(Obj.MetricsOption1)):
            ThisCondition = dict()
            ThisCondition["MetricsOption1"] = Obj.MetricsOption1[idx]
            ThisCondition["MetricsCondition1"] = Obj.MetricsCondition1[idx]
            ThisCondition["DayOffset1"] = Obj.DayOffset1[idx]
            ThisCondition["MetricsOption2"] = Obj.MetricsOption2[idx]
            ThisCondition["MetricsCondition2"] = Obj.MetricsCondition2[idx]
            ThisCondition["DayOffset2"] = Obj.DayOffset2[idx]
            ThisCondition["Operator"] = Obj.Operator[idx]
            ThisCondition["Color"] = Obj.LabelColor[idx]
            ThisCondition["AndOr"] = Obj.AndOr[idx]            
            Conditions.append(ThisCondition)
        
        AndOrList = list()

        for Condition in Conditions:
            Option1 = None
            Option2 = None
            Metric1 = Condition["MetricsOption1"].currentText()
            if Condition["MetricsCondition1"].isEnabled():
                Option1 = Condition["MetricsCondition1"].currentText()
            else:
                Metric1 = Metric1[-3:]

            if Condition["DayOffset1"].isEnabled():
                Offset1 = Condition["DayOffset1"].value()
            else:
                Offset1 = None

            Metric2 = Condition["MetricsOption2"].currentText()
            if Condition["MetricsCondition2"].isEnabled():
                Option2 = Condition["MetricsCondition2"].currentText()
            else:
                Metric2 = Metric2[-3:]

            if Condition["DayOffset2"].isEnabled():
                Offset2 = Condition["DayOffset2"].value()
            else:
                Offset2 = None

            Datum1 = self.getData(Metric1,Option = Option1)
            OriData1 = Datum1.copy()
            Datum2 = self.getData(Metric2,Option = Option2)
            OriData2 = Datum2.copy()
            [Data1,Data2] = self.shiftData([Datum1,Datum2],[Offset1,Offset2])
            #print(OriData1)
            #print(OriData2)
            Operator = Condition["Operator"].currentText()

            Colors.append(self.MainWin.ColorList[Condition["Color"].currentIndex()])

            ThisFoundPoints = self.compareData(Data1,Data2,Operator,Offset1+Offset2)
        
            if Condition["AndOr"].isEnabled():
                AndOrList.append(Condition["AndOr"].currentText())

            if Status == "Out":
                points = list(set(ConditionPoints[0]).intersection(ThisFoundPoints))
                FoundPoints.append(points)
            else:
                FoundPoints.append(ThisFoundPoints)

            PriceList.append(self.getPrice(FoundPoints,{"Datum1":OriData1,"Offset1":Offset1,"Datum2":OriData2,"Offset2":Offset2,"In/Out":Status}))
            
        if len(Conditions) > 1:
            FoundPoints = self.mergeConditions(FoundPoints,AndOrList)
        
        if self.LabelBox.isChecked():
            self.MainWin.sig_FoundPoints.emit([(fp,color,Status) for fp, color in zip(FoundPoints,Colors)])

        Price = [n for m in PriceList for n in m]

        return [Price,FoundPoints]
    
    def getPrice(self,FoundPoints,Condition):
        #Condition: {"Datum1":,"Offset1":"","Datum2":,"Offset2":,"In/Out":}
        #For In: take the lower price, for Out: take the higher price       
        D1 = Condition["Datum1"][0]
        i1 = Condition["Datum1"][1]
        D2 = Condition["Datum2"][0]
        i2 = Condition["Datum2"][1]
        Off1 = Condition["Offset1"]
        Off2 = Condition["Offset2"]
        InOut = Condition["In/Out"]
        AllPrice = list()

        for points in FoundPoints:
            Price = list()
            for pnt in points:
                if pnt > len(D1) or pnt > len(D2):
                    break
                match InOut:
                    case "In":
                        value = D1[pnt+Off1]
                    case "Out":
                        #print(self.MainWin.Metric.Date[pnt+Off1],pnt-Off1,i1[pnt+Off1],D1[pnt+Off1],pnt+Off2,i2[pnt+Off2],D2[pnt+Off2])
                        value = max(D1[pnt+Off1],D2[pnt+Off2])
                    case _:
                        pass
                Price.append([pnt,value])
            AllPrice.append(Price)
        return AllPrice
    
    def mergeConditions(self,FoundPoints,AndOrList):
        # merge the found points depends on selected and/or
        # exclusive "or": the second list will exclude what already existing in the first list
        OutputPoints = list()
        for idx,andor in enumerate(AndOrList):
            fp1 = FoundPoints[idx]
            match andor:
                case "and":
                    fp2 = FoundPoints[idx+1]
                    OutputPoints.append(list(set(fp1).intersection(fp2)))
                case "or":
                    if idx == 0:
                        OutputPoints.append(fp1)
                    else:    
                        OutputPoints.append([pnt for pnt in fp1 if pnt not in OutputPoints[-1]])
                case _:
                    pass
        
        if andor == "or":
            OutputPoints.append([pnt for pnt in FoundPoints[-1] if pnt not in OutputPoints[-1]])

        return OutputPoints            

    def getRange(self):
        self.TableOutput.setDisabled(False)
        [self.BuyPrice,InPoints] = self.pickDataPoint(self.ConditionIn,"In")
        [self.SellPrice,OutPoints] = self.pickDataPoint(self.ConditionOut,"Out", ConditionPoints = InPoints)

        #print(self.InPoints)
        #print(self.OutPoints)
    
    def shiftData(self,data:list,Offset:list):
        Data = data.copy()
        for idx,offset in enumerate(Offset):
            if offset != 0:
                for ind in [0,1]:
                    Data[idx][ind] = Data[idx][ind][:offset]
                    Data[1-idx][ind] = Data[1-idx][ind][-1*offset:]     
        return Data
    
    def compareData(self,data1,data2,op,Offset):
        

        """
        d0 = max(data1[1][0],data2[1][0])
        if data1[1][0] == d0:
            offset = np.where(data2[1]==d0)[0][0]
            data2[0] = data2[0][offset:]
            data2[1] = data2[1][offset:]
        else:
            offset = np.where(data1[1]==d0)[0][0]
            data1[0] = data1[0][offset:]
            data1[1] = data1[1][offset:]
        """
        match op:
            case ">":
                ShiftData = data1[0] - data2[0]
                points = np.where(ShiftData > 0)[0]
            case "<":
                ShiftData = data2[0] - data1[0]
                points = np.where(ShiftData > 0)[0]
            case "=":
                ShiftData = data2[0] - data1[0]
                points = np.where(ShiftData == 0)[0]
            case _:
                pass

        """Get crossover"""
        #data = np.multiply(np.insert(ShiftData,0,0),np.append(ShiftData,0))
        #points = np.where(data < 0)[0]            
        """"""
        #CrossPosition = self.calcCrossPoint(points,data1,data2)
        #Points.append([pnt[1] for pnt in CrossPosition if pnt[0] < data1[1].shape[0] and ShiftData[pnt[0]] > 0])
        Points = [pnt-Offset for pnt in points.tolist() if pnt < data1[1].shape[0]]

        return Points

    def calcCrossPoint(self,points,data1,data2):
        Points = list()
        for pnt in points.tolist():
            db = data1[0][pnt]-data1[0][pnt-1]
            da = data1[1][pnt-1]-data1[1][pnt]
            e = db*data1[1][pnt]+da*data1[0][pnt]
            dd = data2[0][pnt]-data2[0][pnt-1]
            dc = data2[1][pnt-1]-data2[1][pnt]
            f = dd*data2[1][pnt]+dc*data2[0][pnt]
            M = np.array([[db,da],[dd,dc]])
            V = np.array([e,f])
            S = np.dot(np.linalg.inv(M),V)
            Points.append([pnt,S[0]])    
        return Points
                
    def getData(self,text,Option = None):
        match text:
            case "日線":
                DateSN = np.arange(len(self.MainWin.Metric.PackedRaw))
                Data = self.MainWin.DataSource.PeriodData
                match Option:
                    case "開盤":
                        Data = Data["Open"]
                    case "收盤":
                        Data = Data["Close"]
                    case "最高":
                        Data = Data["High"]
                    case "最低":
                        Data = Data["Low"]
                    case _:
                        pass
                Data = [Data.values,DateSN]

            case "日均線":
                Data = self.MainWin.Metric.RollingAvg
            case "RSI":
                pass
            case _:
                pass
        
        return Data
    
class MainWin(QtWidgets.QWidget):
    sig_FoundPoints = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("富寬肉燥飯")
        self.ColorList = ["red","blue","green","white","purple","cyan","orange"]
        self.MetricList = ["日線","日均線","RSI"]

        self.DataSource = DataSourceGroup(self)
        self.Metric = MetricGroup(self)
        self.Condition = ConditionGroup(self)
        self.Metric.setDisabled(True)
        self.Condition.setDisabled(True)

        self.Layout = QtWidgets.QGridLayout()
        self.Layout.addWidget(self.DataSource,0,0,2,2)
        self.Layout.addWidget(self.Metric,2,0,2,2)
        self.Layout.addWidget(self.Condition,4,0,2,2)
        self.setLayout(self.Layout)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWin()
    win.show()
    sys.exit(app.exec_())

    #Saya Fujiwara
    #Kuru Shichisei
    #Rina Nanase
    #Mimi
    #Suzu Ichinose
    #kawd00793
    #Yui Oba