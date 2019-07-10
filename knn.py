from preprocess import *
from quadTree import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from queue import PriorityQueue
from time import time


class KnnElement:
    '''为knn中优先队列准备的元素'''

    def __init__(self, root: Node, minDist: float):
        self.root = root
        self.minDist = minDist

    def __lt__(self, other):
        return self.minDist < other.minDist

class KnnElementReverse:
    '''为knn中优先队列准备的元素'''

    def __init__(self, index: int, dist: float):
        self.index = index
        self.dist = dist

    def __lt__(self, other):
        return self.dist > other.dist

def forceKnn(point: list, K: int, pointSet: list, dataSetDict: dict) -> list:
    maxScore = 0
    res = PriorityQueue()
    for i in pointSet:
        tempPoint = dataSetDict[i]
        dist = euclidianDistance(point, tempPoint)
        tempNode = KnnElementReverse(i, dist)
        if K > 0:
            res.put(tempNode)
            K = K - 1
        else:
            temp = res.get()
            if tempNode.dist < temp.dist:
                res.put(tempNode)
            else:
                res.put(temp)
    
    resList = []
    while not res.empty():
        temp = res.get()
        resList.append(temp.index)

    return resList


def knn(point: list, K: int, tree: QuadTree, dataSetDict: dict) -> list:
    maxScore = 0
    qu = PriorityQueue()
    qu.put(KnnElement(tree.root, 0))    # 向队列增加一个元素 距离最短为0
    res = PriorityQueue()
    while not qu.empty():
        currentNode = qu.get()
        if K == 0 and minDist(point, currentNode.root.rect) > maxScore:
            break
        elif len(currentNode.root.children) == 0:    # 如果这个节点是叶节点
            for i in currentNode.root.points:
                tempPoint = dataSetDict[i]
                dist = euclidianDistance(point, tempPoint)
                tempNode = KnnElementReverse(i, dist)
                if K > 0:                   # 没有满直接放入
                    res.put(tempNode)
                    maxScore = max(maxScore, dist)
                    K = K - 1
                else:   # 否则判断是否需要更新
                    temp = res.get()
                    if tempNode.dist < temp.dist:
                        res.put(tempNode)
                    else:
                        res.put(temp)
                    temp = res.get()                  
                    maxScore = max(maxScore, temp.dist)
                    res.put(temp)
        else:
            for child in currentNode.root.children:
                qu.put(KnnElement(child, minDist(point, child.rect)))

    resList = []
    while not res.empty():
        temp = res.get()
        resList.append(temp.index)
    return resList


#判断一个点是否已经遍历过了
def traversed(point:int,hasTraversed:list):
    for p in hasTraversed:
        if p == point:
            return True
    return False


def createTraKNN(point: list, K: int, tree: QuadTree, dataSetDict: dict,hasTraversed:list) -> list:
    maxScore = 0
    qu = PriorityQueue()
    qu.put(KnnElement(tree.root, 0))    # 向队列增加一个元素 距离最短为0
    res = []
    while not qu.empty():
        currentNode = qu.get()
        if K == 0 and minDist(point, currentNode.root.rect) > maxScore:
            break
        elif len(currentNode.root.children) == 0:    # 如果这个节点是叶节点
            for i in currentNode.root.points:
                #如果已经遍历过，直接跳过
                if traversed(i,hasTraversed):
                    continue

                tempPoint = dataSetDict[i]
                dist = euclidianDistance(point, tempPoint)
                if K > 0:                   # 没有满直接放入
                    res.append(i)
                    maxScore = max(maxScore, dist)
                    K = K - 1
                elif dist < maxScore:    # 如果有更小的结果 更新数组
                    for j in range(len(res)):
                        if euclidianDistance(dataSetDict[res[j]], point) == maxScore:
                            res[j] = i
                            maxScore = dist
                            break
                    for j in res:
                        maxScore = max(euclidianDistance(dataSetDict[j], point), maxScore)
        else:
            for child in currentNode.root.children:
                qu.put(KnnElement(child, minDist(point, child.rect)))
    
    return res


# 显示所有的点
def show():
    plt.figure(1)
    dataSet = preprocess.readDataSet()
    dataSetDict = preprocess.readDataSetAsDict()
    x, y = dataSet[:, 0], dataSet[:, 1]
    plt.scatter(x, y, s=1)


# 将KNN和forceKNN结果进行比较以判断是否正确
def testResultIsRight():
    plt.figure(1)
    print("Test Whether the result is correct")
    dataSetDict = preprocess.readDataSetAsDict()
    tree = QuadTree(dataSetDict, 500)
    # 目标点
    tarPoint = [33, -117.5]
    pointNum = 20
    print("The tarPoint:",tarPoint,"aim to find {} NN points".format(pointNum))
    print("In case of Force KNN:")
    res = forceKnn(tarPoint, pointNum, range(len(dataSetDict)), dataSetDict)
    res.sort()
    print(res)

    print("In case of Quad-Tree KNN:")
    res = knn(tarPoint, pointNum, tree, dataSetDict)
    res.sort()
    print(res)
    drawAllRect(tree.root)

    points = []
    for i in res:
        points.append(dataSetDict[i])
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    plt.scatter(x, y, s=30, c='g')
    plt.scatter([tarPoint[0]], [tarPoint[1]], s=50, c='y')
    plt.axis('equal')


# 检测不同情况下的平均寻找时间
def testDifferentCases():
    plt.figure(2)
    dataSetDict = preprocess.readDataSetAsDict()
    tarPoint = [33, -117.5]
    times = 10  # 每种情况找10次取平均值
    x = range(10, 200, 10) # K的取值范围
    # 暴力的时间随K范围变化的时间复杂度
    force = []
    pointSet = range(len(dataSetDict))
    for i in x:
        s = time()
        for j in range(times):
            forceKnn(tarPoint, i, pointSet, dataSetDict)
        e = time()
        force.append((e-s)/times)
    
    plt.plot(x, force, color="black", label="Force Search")
    
    # 基于不同四叉树的不同K的寻找时间
    # 根据四叉树实验 选取5个不同深度的树
    capacity = [10, 50, 100, 600, 1600, 3600]
    colors = ["red", "blue", "yellow", "green", "pink", "orange"]
    for index in range(len(capacity)):
        res = []    # 存放不同树的耗费时间
        tree = QuadTree(dataSetDict, capacity[index])
        for i in x:
            s = time()
            for j in range(times):
                knn(tarPoint, i, tree, dataSetDict)
            e = time()
            res.append((e-s)/times)
        plt.plot(x, res, color=colors[index], label=str(capacity[index]))

    plt.xlabel("Search Points' Number")
    plt.ylabel("Average Time")
    plt.legend()
        

def main():
    if not os.path.exists(preprocess.dataSetPath) and not os.path.exists(preprocess.newDataSetPath):
        print("Error:","dataSetPath","not exist")
        return
    if not os.path.exists(preprocess.newDataSetPath):
        preprocess.writeDataSetToFile()
    
    # 展示一个样例
    show()
    # 显示结果是否正确，和暴力比较
    testResultIsRight()
    # 检测不同的四叉树KNN算法和暴力之间的比较
    # 耗费时间比较长 默认注释
    # testDifferentCases()
    plt.show()

if __name__ == "__main__":
    main()