import preprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import os
from time import time


# 一些辅助工具函数
def euclidianDistance(a: list, b: list) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def minRect(pointSet: list, dataSetDict: dict) -> list:
    ''' 求解外包最小矩形 根据点的下标集合和数据字典 '''
    point1 = dataSetDict[pointSet[0]].copy()  # 左上角
    point2 = dataSetDict[pointSet[0]].copy()  # 右下角
    for i in pointSet:
        point1[0] = dataSetDict[i][0] if dataSetDict[i][0] < point1[0] else point1[0]
        point1[1] = dataSetDict[i][1] if dataSetDict[i][1] > point1[1] else point1[1]
        point2[0] = dataSetDict[i][0] if dataSetDict[i][0] > point2[0] else point2[0]
        point2[1] = dataSetDict[i][1] if dataSetDict[i][1] < point2[1] else point2[1]

    return [point1[0], point1[1], point2[0] - point1[0], point1[1] - point2[1]]


def minDist(point: list, rect: list) -> int:
    ''' 求解点到矩形的距离 '''
    Px, Py = point[0], point[1]
    x, y, w, h = rect

    # 当点在矩形内则返回距离0
    if Px >= x and Px <= x+w and Py <= y and Py >= y-h:
        return 0
    # 在x轴之间
    elif Px >= x and Px <= x+w:
        return min(abs(Py - y), abs(Py - y + h))
    # 在y轴之间
    elif Py <= y and Py >= y-h:
        return min(abs(Px - x), abs(Px - x - w))

    x1, x2, y1, y2 = x, x + w, y, y - h
    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    res = 0xffffffff
    for p in points:
        res = min(res, ((p[0]-Px)**2 + (p[1]-Py)**2)**0.5)

    return res


def isInRect(point: list, rect: list) -> bool:
    '''判断一个点是否在一个矩形内'''
    if minDist(point, rect) == 0:
        return True
    else:
        return False


def drawRect(rect: list):
    xy, w, h = [rect[0], rect[1]-rect[3]], rect[2], rect[3]
    plt.gca().add_patch(mp.Rectangle(xy, w, h, edgecolor='r', facecolor='none'))


class Node:
    '''四叉树节点类:
    Attributes:
        rect: 节点区域大小依次为：左上角顶点坐标xy，宽度，高度
        pointsNum: 区域内的节点总数
        points：叶节点存储真正的点
        children：儿子节点（四个）
    '''

    def __init__(self, rect: list, pointsNum: int):
        self.rect = rect
        self.pointsNum = pointsNum
        self.points = []
        self.children = []


class QuadTree:
    '''四叉树类:
    Attributes:
        capacity: 每个窗口的最小密度
        root: 四叉树的根节点
    '''

    def creatTree(self, rect: list, pointSet: list, dataSetDict: dict) -> Node:
        '''根据一个窗口和一组数据返回一个四叉树
        Args:
            rect: 需要建立四叉树的窗口
            pointSet：此窗口中的数据集 存储为点的下标形式
            dataSetDict：真正的数据集字典
        Return:
            root：一个四叉树的根节点
        '''
        pointsNum = len(pointSet)
        tempRoot = Node(rect, len(pointSet))
        if pointsNum <= self.capacity:
            tempRoot.points = pointSet
        else:
            # 获取大的区域的四等分子区域和子区域的点集
            subSpace, subPoints = self.splitSpace(rect, pointSet, dataSetDict)
            for i in range(4):
                tempRoot.children.append(self.creatTree(subSpace[i], subPoints[i], dataSetDict))

        return tempRoot

    def splitSpace(self, rect: list, pointSet: list, dataSetDict: dict) -> list:
        '''将父窗口切割为四个小窗口同时切分数据
        Args:
            rect：父窗口大小
            pointSet：此窗口中的数据集 存储为点的下标形式
            dataSetDict：真正的数据集字典
        Return：
            subSpace：四个小区域的大小和位置
            subPoints：四个小区域的数据 存储点的下标
        '''
        x, y, w, h = rect
        w, h = w/2, h/2
        subSpace = [[x, y, w, h],
                    [x + w, y, w, h],
                    [x, y - h, w, h],
                    [x + w, y - h, w, h]]

        subPoints = [[], [], [], []]
        for i in pointSet:
            index = 0
            if isInRect(dataSetDict[i], subSpace[0]):
                index = 0
            elif isInRect(dataSetDict[i], subSpace[1]):
                index = 1
            elif isInRect(dataSetDict[i], subSpace[2]):
                index = 2
            else:
                index = 3
            subPoints[index].append(i)
        return subSpace, subPoints

    def __init__(self, dataSetDict: dict, capacity: int):
        self.capacity = capacity
        pointSet = range(len(dataSetDict))
        self.root = self.creatTree(minRect(pointSet, dataSetDict), pointSet, dataSetDict)


def drawAllRect(root):
    drawRect(root.rect)
    for i in root.children:
        drawAllRect(i)


def getTreeHeight(root):
    res = 0
    for i in root.children:
        res = max(res, 1 + getTreeHeight(i))
    return res


# 在图上显示一个具体的四叉树
def show():
    plt.figure(1)
    dataSetDict = preprocess.readDataSetAsDict()
    dataSet = preprocess.readDataSet()
    preprocess.drawPoints(dataSet)
    # 500 capacity 展示
    quadTree = QuadTree(dataSetDict, 500)
    drawAllRect(quadTree.root)
    plt.axis('equal')
    plt.title("Quad-Tree With Capacity 500")


# 不同capacity花费构造时间比较
def test():
    plt.figure(2)
    dataSetDict = preprocess.readDataSetAsDict()
    x = list(range(100,6000,500))
    y = []
    print("capacity\taverage-time\t\theight")
    for i in range(100, 6000, 500):
        times = 5
        s = time()
        for j in range(times):
            quadTree = QuadTree(dataSetDict, i)
        e = time()
        height = getTreeHeight(quadTree.root)
        print("{}\t\t{}\t{}".format(i, (e-s)/times, height))
        y.append((e-s)/times)
    
    plt.plot(x, y, 'r-o')
    plt.xlabel("Capacity")
    plt.ylabel("Average Time")


# 性能测试和可视化
def main():
    if not os.path.exists(preprocess.dataSetPath) and not os.path.exists(preprocess.newDataSetPath):
        print("Error:","dataSetPath","not exist")
        return
    if not os.path.exists(preprocess.newDataSetPath):
        preprocess.writeDataSetToFile()
    # 使用matlibplot展示一个样例
    show()
    # test耗费时间过长 默认注释掉
    # 取消注释查看不同情况的四叉树生成时间
    # test()
    plt.show()


if __name__ == "__main__":
    main()
