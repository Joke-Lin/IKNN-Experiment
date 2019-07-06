import numpy as np
import matplotlib.pyplot as plt

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

def readDataSet()->np.ndarray:
    ''' 从上面定义的numpy型数据集读取数据并返回 '''
    return np.load(newDataSetPath)

def visualization():
    ''' 简单的可视化处理 '''
    writeDataSetToFile()
    dataSet = readDataSet()
    x, y = dataSet[:,0], dataSet[:,1]
    plt.scatter(x, y, s=1)
    plt.show()

if __name__ == "__main__":
    visualization()
