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

def forceKnn(point: list, K: int, dataSet: np.ndarray) -> list:
    maxScore = 0
    res = []
    for tempPoint in dataSet:
        dist = euclidianDistance(point, tempPoint)
        if K > 0:
            res.append(tempPoint[:])
            maxScore = max(maxScore, dist)
            K = K - 1
        elif dist < maxScore:
            for i in range(len(res)):
                if euclidianDistance(res[i], point) == maxScore:
                    res[i] = tempPoint[:]
                    maxScore = dist
                    break
            for i in res:
                maxScore = max(euclidianDistance(i, point), maxScore)
    return res    


def knn(point: list, K: int, tree: QuadTree) -> list:
    maxScore = 0
    qu = PriorityQueue()
    qu.put(KnnElement(tree.root, 0))    # 向队列增加一个元素 距离最短为0
    res = []
    while not qu.empty():
        currentNode = qu.get()
        if K == 0 and minDist(point, currentNode.root.rect) > maxScore:
            break
        elif len(currentNode.root.children) == 0:    # 如果这个节点是叶节点
            for tempPoint in currentNode.root.points:
                dist = euclidianDistance(point, tempPoint)
                if K > 0:                   # 没有满直接放入
                    res.append(tempPoint)
                    maxScore = max(maxScore, dist)
                    K = K - 1
                elif dist < maxScore:    # 如果有更小的结果 更新数组
                    for i in range(len(res)):
                        if euclidianDistance(res[i], point) == maxScore:
                            res[i] = tempPoint
                            maxScore = dist
                            break
                    for i in res:
                        maxScore = max(euclidianDistance(i, point), maxScore)
        else:
            for child in currentNode.root.children:
                qu.put(KnnElement(child, minDist(point, child.rect)))
    
    return res

def visualization():
    dataSet = preprocess.readDataSet()
    x, y = dataSet[:, 0], dataSet[:, 1]
    plt.scatter(x, y, s=1)
    rect = minRect(dataSet)
    tree = QuadTree(minRect(dataSet), dataSet, 1000)
    tarPoint = [33, -117.5]
    s = time()
    res = knn(tarPoint, 10, tree)
    # drawAllRect(tree.root)
    # dataSet = np.array(res)
    # x, y = res[:, 0], res[:, 1]
    # plt.scatter(x, y, s=30, c='g')
    # plt.scatter([33], [-117.5], s=50, c='y')
    # plt.axis('equal')
    # plt.show()
    e = time()
    dist = [euclidianDistance(tarPoint, x) for x in res]
    dist.sort()
    print(dist)
    print("time:" + str(e-s))
    s = time()
    res = forceKnn(tarPoint, 10, dataSet)
    e = time()
    print("force:")
    dist = [euclidianDistance(tarPoint, x) for x in res]
    dist.sort()
    print(dist)
    print("time:" + str(e-s))


if __name__ == "__main__":
    visualization()