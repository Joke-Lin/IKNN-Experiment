# -*-coding:utf8 -*-
import numpy as np
import random
import queue
import matplotlib.pyplot as plt

from quadTree import *
import preprocess
from  knn import *
import os

outfilep2tra = r'./data-set/point2trajectory.npy'
outfiletra2p = r'./data-set/trajectory2point.npy'

#生成轨迹
def createTrajectories()->list:
    #数据集作为字典保存
    datasetDict = preprocess.readDataSetAsDict()
    tree = QuadTree(datasetDict,100)

    #10个点一条轨迹
    K = 10
    #生成5000条轨迹
    trajectoryNum = 5000
    
    #最后返回的轨迹的经纬度列表
    res = []
    #轨迹到点集的字典
    trajectory2Points = {}
    #点到轨迹集的字典
    points2Trajectory = {}

    for i in range(trajectoryNum):
        #随机生成一个下标
        index = random.randint(0,tree.root.pointsNum-1)

        #返回一个轨迹的点列表
        trajectory =  createOneTrajectory(tree,index,K,datasetDict)
        trajectory = sorted(trajectory,key = lambda point:datasetDict[point][1])

        #将轨迹加入到相应轨迹字典
        trajectory2Points[i] = trajectory

        #遍历轨迹，将点对应的轨迹记录下来
        for p in trajectory:
            #如果字典中有这个键，则直接将轨迹下标加入即可
            if p in points2Trajectory:
                points2Trajectory[p].append(i)
            else:#否则初始化列表
                points2Trajectory[p] = [i]
        res.append(trajectory)

    return [res,trajectory2Points,points2Trajectory,datasetDict,tree]

#生成一条轨迹
def createOneTrajectory(tree:QuadTree,point:int,K:int,datasetDict:dict)->list:
    #保存已经找到的点
    hasTraversed = [point]

    points = findNextNearst2Points(tree,point,hasTraversed,datasetDict)
    point = points[0]
    hasTraversed.append(point)

    #再找K-2个点
    for i in range(K-2):
        
        #保证了找到的两个点都没有出现过
        points = findNextNearst2Points(tree,point,hasTraversed,datasetDict)
        index = random.randint(0,1)
        point = points[index]
        hasTraversed.append(point)
    
    return hasTraversed


#查找一个点最近的两个点
def findNextNearst2Points(tree:QuadTree,point:int,hasTraversed:list,datasetDict:dict)->list:
    points = createTraKNN(datasetDict[point],2,tree,datasetDict,hasTraversed)
    return points

#遍历四叉树，输出每一个节点的点的个数
def traverseQuadTree(root:Node):
    temproot = root
    if not temproot:
        return 

    for onechild in temproot.children:
        if onechild.children:
            traverseQuadTree(onechild)
        else:
            print(onechild.pointsNum)

def readTrajectory():
    res = createTrajectories()
    trajectories = res[0]
    datasetDict = res[3]
    onetrajectory = trajectories[0]
    onetrajectory = sorted(onetrajectory,key = lambda point:datasetDict[point][1])
    x = []
    y = []
    for p in onetrajectory:
        x.append(datasetDict[p][1])
        y.append(datasetDict[p][0])
    plt.subplot(1,2,1)
    plt.plot(x,y)
    plt.subplot(1,2,2)
    plt.scatter(x,y)
    plt.show()

    print('轨迹到点')
    t1 = res[1][0]
    P = 0
    for p in t1:
        print(p)
        P = p
    
    print('点到轨迹')
    p1 = res[2][P]
    print('changdu:')
    print(len(p1))
    for ti in p1:
        print(ti)

#生成轨迹并将 相应的 索引表通过numpy 保存
def save():
    res = createTrajectories()
    tra2point = res[1] #轨迹到点的映射
    point2tra = res[2] #点到轨迹的映射

    #保存轨迹到点的映射
    array1 = []
    for key,value in tra2point.items():
        value.append(key)
        array1.append(value)
    array1 = np.array(array1)
    np.save(outfiletra2p,array1)

    #保存点到轨迹的映射
    array2 = []
    for key,value in point2tra.items():
        value.append(key)
        array2.append(value)
    array2 = np.array(array2)
    np.save(outfilep2tra,array2)

#加载数据集：点集,轨迹集,点经过轨迹集
#生成四叉树
#以列表形式返回
def load():
    datasetDict = preprocess.readDataSetAsDict()
    tree = QuadTree(datasetDict,50)
    tra2point = np.load(outfiletra2p)
    point2tra = np.load(outfilep2tra)
    tra2pDict = dict()
    p2traDict = dict()

    for t in tra2point:
        points = list(t)
        key = points.pop()
        tra2pDict[key] = points
    
    for p in point2tra:
        tra = list(p)
        key = tra.pop()
        p2traDict[key] = tra

    return [tra2pDict,p2traDict,tree,datasetDict]


def main():
    if not os.path.exists(preprocess.dataSetPath) and not os.path.exists(preprocess.newDataSetPath):
        print("Error:","dataSetPath","not exist")
        return

    if not os.path.exists(preprocess.newDataSetPath):
        preprocess.writeDataSetToFile()

    if not os.path.exists(outfilep2tra) or not os.path.exists(outfiletra2p):
        save()

    tra2pDict,p2traDict,tree,datasetDict = load()
    dataSet = preprocess.readDataSet()
    preprocess.drawPoints(dataSet)
    for points in tra2pDict.values():
        temp = [datasetDict[x] for x in points]
        temp = np.array(temp)
        x, y = temp[:, 0], temp[:, 1]
        plt.plot(x, y, linewidth=1)
    
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()