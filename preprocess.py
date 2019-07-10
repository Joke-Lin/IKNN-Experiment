import numpy as np
import matplotlib.pyplot as plt
import os

dataSetPath = "./data-set/la_points.txt"        # 原始文本型数据集路径
newDataSetPath = "./data-set/la_points.npy"     # 新的二进制型数据集路径


def writeDataSetToFile():
    '''
    将最初的文本数据转换为二进制数据，格式为numpy.ndarray
    保存在文件：./data-set/la_points.npy 中
    '''
    with open(dataSetPath, "r") as f:
        dataSet = []
        dataSetStr = f.read()
        for eachLine in dataSetStr.split():
            tempPoint = eachLine.split(',')
            point = [float(tempPoint[0]), float(tempPoint[1])]
            dataSet.append(point)
        dataSet = np.array(dataSet)
        np.save(newDataSetPath, dataSet)


def readDataSet() -> np.ndarray:
    ''' 从上面定义的numpy型数据集读取数据并返回 '''
    return np.load(newDataSetPath)


def readDataSetAsDict() -> dict:
    dataSetDict = {}
    dataSet = readDataSet()
    i = 0
    for point in dataSet:
        dataSetDict[i] = list(point)
        i += 1
    
    return dataSetDict


def drawPoints(dataSet):
    x, y = dataSet[:, 0], dataSet[:, 1]
    plt.scatter(x, y, s=1)


def main():
    ''' 简单的可视化处理 '''
    if not os.path.exists(dataSetPath) and not os.path.exists(newDataSetPath):
        print("Error:","dataSetPath","not exist")
        return
    if not os.path.exists(newDataSetPath):
        writeDataSetToFile()
    # 读入数据
    dataSet = readDataSet()
    # 开始绘制
    drawPoints(dataSet)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
