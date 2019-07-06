import preprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp


# 一些辅助工具函数
def minRect(dataSet: np.ndarray) -> list:
    ''' 求解外包最小矩形 '''
    point1 = dataSet[0].copy()  # 左上角
    point2 = dataSet[0].copy()  # 右下角
    for point in dataSet:
        point1[0] = point[0] if point[0] < point1[0] else point1[0]
        point1[1] = point[1] if point[1] > point1[1] else point1[1]
        point2[0] = point[0] if point[0] > point2[0] else point2[0]
        point2[1] = point[1] if point[1] < point2[1] else point2[1]

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

    def creatTree(self, rect: list, dataSet: np.ndarray) -> Node:
        '''根据一个窗口和一组数据返回一个四叉树
        Args:
            rect: 需要建立四叉树的窗口
            dataSet：此窗口中的数据集
        Return:
            root：一个四叉树的根节点
        '''
        drawRect(rect)
        pointsNum = len(dataSet)
        tempRoot = Node(rect, len(dataSet))
        if pointsNum <= self.capacity:
            tempRoot.points = dataSet
        else:
            # 获取大的区域的四等分子区域和子区域的点集
            subSpace, subPoints = self.splitSpace(rect, dataSet)
            for i in range(4):
                tempRoot.children.append(
                    self.creatTree(subSpace[i], subPoints[i]))

        return tempRoot

    def splitSpace(self, rect: list, dataSet: np.ndarray) -> list:
        '''将父窗口切割为四个小窗口同时切分数据
        Args:
            rect：父窗口大小
            dataSet：窗口内数据
        Return：
            subSpace：四个小区域的大小和位置
            subPoints：四个小区域的数据
        '''
        x, y, w, h = rect
        w, h = w/2, h/2
        subSpace = [[x, y, w, h],
                    [x + w, y, w, h],
                    [x, y - h, w, h],
                    [x + w, y - h, w, h]]

        subPoints = [[], [], [], []]
        for point in dataSet:
            index = 0
            if isInRect(point, subSpace[0]):
                index = 0
            elif isInRect(point, subSpace[1]):
                index = 1
            elif isInRect(point, subSpace[2]):
                index = 2
            else:
                index = 3
            subPoints[index].append(point)
        return subSpace, subPoints

    def __init__(self, rect: list, dataSet: np.ndarray, capacity: int):
        self.capacity = capacity
        self.root = self.creatTree(rect, dataSet)


def visualization():
    dataSet = preprocess.readDataSet()
    x, y = dataSet[:, 0], dataSet[:, 1]
    plt.scatter(x, y, s=1)
    rect = minRect(dataSet)
    quadTree = QuadTree(minRect(dataSet), dataSet, 500)
    plt.show()


if __name__ == "__main__":
    visualization()
