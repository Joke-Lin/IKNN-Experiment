# -*-coding:utf8 -*-
import numpy as np
import math
import random
import queue
import matplotlib.pyplot as plt

from quadTree import *
import preprocess
from  knn import *
from createTrajectories import *



#查找每一个查询点的K临近点
def findKNN(Q:list,tree:QuadTree,K:int,datasetDict:dict):
    res = []
    for qi in Q:
        #查找最临近的K个点
        resi = knn(datasetDict[qi],K,tree,datasetDict)
        res.append(resi)
    return res

#随机生成一个查询序列,为临近的K个点
def generateQuery(K:int, tree:QuadTree,datasetDict:dict)->list:
    begin = random.randint(0,tree.root.pointsNum-1)
    #找最临近的另外两个点 加上初始点 作为查询点集合
    res = knn(datasetDict[begin],K,tree,datasetDict)
    return res


#查找查询点KNN点经过的轨迹
#第一个参数为列表的列表,每一个列表对应一个查询点qi的KNN点集
def findTrajectroy(points:list,point2tra:dict)->set:
    #最后的所有轨迹并集
    res = set()
    for pp in points:#每一个查询点查找到的KNN点集
        for p in pp: #每一个KNN点
            #点不经过任何轨迹则跳过
            if p not in point2tra:
                continue
            for ti in point2tra[p]:#点所经过的轨迹
                res.add(ti)
    return res


#计算已经扫描的轨迹的上界
#t:当前轨迹的key,points:查询点对应的KNN点集列表
def UB(t:int,tra2point:dict,point2tra:dict,Query:list,points:list,datasetDict:dict)->float:
    sum = 0
    #对应查询点查询到的KNN点集
    for index in range(len(Query)):
        qi = Query[index]
        pp = points[index]
        mindistance = 0xffffffff
        for p in pp:#对每一个K临近点，判断是否经过t这条轨迹,如果经过则更新最短距离
            #如果该点未经过任何轨迹则跳过
            if p not in point2tra:
                continue
            #判断点是否经过当前轨迹
            if isExist(point2tra[p],t):
                curdist = euclidianDistance(datasetDict[p],datasetDict[qi])
                if curdist < mindistance:
                    mindistance = curdist
        #如果没有点经过当前轨迹,则计算距离最大的作为当前sim(qi,Rx),则当前的值一定大于真实值
        #故为上界
        if mindistance == 0xffffffff:
            maxdistance = -100
            for p in pp:#对每一个K临近点,直接算出其中距离最远的
                curdist = euclidianDistance(datasetDict[p],datasetDict[qi])
                if curdist > maxdistance:
                    maxdistance = curdist
            sum += math.exp(-maxdistance)
        else:
            sum += math.exp(-mindistance)
    return sum    

#计算未扫描的轨迹的上界
def UBn(Query:list,points:list,datasetDict:dict)->float:
    sum = 0
    #对应查询点查询到的KNN点集
    for index in range(len(Query)):
        qi = Query[index]
        pp = points[index]
        maxdistance = -100
        for p in pp:#对每一个K临近点,直接算出其中距离最远的
            curdist = euclidianDistance(datasetDict[p],datasetDict[qi])
            if curdist > maxdistance:
                    maxdistance = curdist
        sum += math.exp(-maxdistance)
    return sum    

#计算一条轨迹对于查询点的lower bound 
#参数分别为:轨迹下标，轨迹到点的映射，点到轨迹的映射，查询点集，查询点的KNN集合
def LB(t:int,tra2point:dict,point2tra:dict,Query:list,points:list,datasetDict:dict)->float:
    sum = 0
    #对应查询点查询到的KNN点集
    for index in range(len(Query)):
        qi = Query[index]
        pp = points[index]
        mindistance = 0xffffffff
        for p in pp:#对每一个K临近点，判断是否经过t这条轨迹,如果经过则更新最短距离
            #如果该点未经过任何轨迹直接跳过
            if p not in point2tra:
                continue
            #判断点是否经过当前轨迹
            if isExist(point2tra[p],t):
                curdist = euclidianDistance(datasetDict[p],datasetDict[qi])
                if curdist < mindistance:
                    mindistance = curdist
        #如果没有点经过当前轨迹,则不作改动
        if mindistance == 0xffffffff:
            sum += 0
        else:
            sum += math.exp(-mindistance)
    return sum

#判断一个轨迹集合是否存在当前轨迹
def isExist(T:list,t:int)->bool:
    for ti in T:
        if ti == t:
            return True
    return False



#用于refine 函数 使用的元素类型
class K_BCT_element():
    def __init__(self,t:int,Sim:float):
        self.sim = Sim
        self.t = t
    def __lt__(self,other):
        return self.sim < other.sim    

#一组查询点 和 一条轨迹相似度
def Similarity(Q:list,T:list,datasetDict:dict):
    sum = 0
    for q in Q:
        mindist = 0xffffffff
        for pi in T:
            curdist = euclidianDistance(datasetDict[q],datasetDict[pi])
            if curdist < mindist:
                mindist = curdist
        sum += math.exp(-mindist)
    return sum

#通过准确计算给出最终的结果集合
#C:待定的轨迹集合，
def refine(C:list,K:int,tra2point:dict,point2tra:dict,Query:list,points:list,datasetDict:dict)->list:
    # print('Candidates length:',len(C))
    res = []
    Candidates = []

    #计算每一条轨迹的上界
    for t in C:
        Sim = UB(t,tra2point,point2tra,Query,points,datasetDict)

        #加入到待定集合Candidates 方便后续排序使用
        Candidates.append(K_BCT_element(t,Sim))
    
    #按照UB 降序排序    
    Candidates = sorted(Candidates,key = lambda ele:ele.sim,reverse = True)

    #遍历Candidates, UB 从大到小
    for i in range(len(Candidates)):
        #计算准确地相似度并更新
        Sim_i = Similarity(Query,tra2point[Candidates[i].t],datasetDict)
        Candidates[i].sim = Sim_i

        #前K个轨迹不需要考虑
        if i < K:
            res.append(Candidates[i])
        else:
            #将备选的K个结果按升序排序,如果当前相似度大于了被选集中最小的则需要进行更新。
            res.sort(key = lambda ele:ele.sim)
            if Sim_i > res[0].sim:
                res[0] = Candidates[i]
                res.sort(key = lambda ele:ele.sim)#再次排序便于后一个判断使用
            
            #如果当前已经是最后一个元素了，或者被备集中最小的相似度 都比 后一个元素的UB大，则不用继续判断(UB按照降序排列的)
            if i == len(Candidates) - 1 or res[0].sim >= Candidates[i+1].sim:
                #将最后的轨迹KEY保存为list 返回
                result = []
                for ele in res:
                    result.append(ele.t)
                return result 
    result = []
    for ele in res:
        result.append(ele.t)
    return result 

#关键的incremental KNN 函数
#输入为查找点列表 和 要找的轨迹条数
def IKNN(Query:list,K:int,point2tra:dict,tra2point:dict,tree:QuadTree,datasetDict:dict)->list:

    #每次增加查找的个数 以及KNN初始值
    delta = len(Query)*K
    y = delta

    while True:
        #查找KNN点集
        points = findKNN(Query,tree,y,datasetDict)
         #计算出待定的轨迹集
        C = findTrajectroy(points,point2tra)

        #对待定轨迹集合进行判断
        if len(C) >= K:
            #待定集合中每条轨迹的下界
            Lowerbound = []
            for t in C:
                Lowerbound.append(  LB(t,tra2point,point2tra,Query,points,datasetDict) )

            #未扫描集合的上界
            Upperbound = UBn(Query,points,datasetDict)

            Lowerbound.sort()
            #用列表切片K个最大的下界值
            k_LB = Lowerbound[-K:]

            #如果最小下界 比 上界都大，则不需要进行比较了
            if k_LB[0] >= Upperbound:
                #进行进一步准确计算得出最后的轨迹集合
                K_BCT = refine(C,K,tra2point,point2tra,Query,points,datasetDict)
                return K_BCT

        #找到的轨迹条数不足K条,增加搜索的范围进一步查找
        y = y + delta


class ForceItem():
    def __init__(self,t:int,Sim:float):
        self.sim = Sim
        self.t = t
    def __lt__(self,other):
        return self.sim < other.sim


# 暴力遍历的 寻找最相似轨迹
def forceIKNN(Query:list, K:int, tra2point:dict, datasetDict:dict)->list:
    res = queue.PriorityQueue()
    for index,tra in tra2point.items():
        sim = Similarity(Query,tra,datasetDict)
        tempNode = ForceItem(index, sim)
        if K > 0:
            res.put(tempNode)
            K = K - 1
        else:
            temp = res.get()
            if tempNode.sim > temp.sim:
                res.put(tempNode)
            else:
                res.put(temp)
    
    resList = []
    while not res.empty():
        resList.append(res.get().t)
    
    return resList


# 展示一个样例
def showOneExample(tra2point, point2tra, tree, datasetDict):
    plt.figure(1)
    dataSet = preprocess.readDataSet()
    preprocess.drawPoints(dataSet)
    for points in tra2point.values():
        temp = [datasetDict[x] for x in points]
        temp = np.array(temp)
        x, y = temp[:, 0], temp[:, 1]
        plt.plot(x, y, linewidth=1, color="silver")
    
    Query = generateQuery(10,tree,datasetDict)
    K = 10
    C = IKNN(Query, K, point2tra, tra2point,tree, datasetDict)
    temp = []
    for i in Query:
        temp.append(datasetDict[i])
    temp = np.array(temp)
    x, y = temp[:, 0], temp[:, 1]
    plt.plot(x, y, linewidth=3,color='black')

    for index in C:
        points = tra2point[index]
        temp = [datasetDict[x] for x in points]
        temp = np.array(temp)
        x, y = temp[:, 0], temp[:, 1]
        plt.plot(x, y, linewidth=2)

    
# 检测算法准确性 和 暴力对比
def testIKNNResultIsRight(tra2point, point2tra, tree, datasetDict):
    K = 10
    Query = generateQuery(10,tree,datasetDict)
    C = forceIKNN(Query, K, tra2point, datasetDict)
    C.sort()
    print(C)

    C = IKNN(Query, K, point2tra, tra2point,tree, datasetDict)
    C.sort()
    print(C)
    

# 检测不同IKNN以及暴力在不同查询数量上面的区别
def testDifferentIKNN(tra2point, point2tra, tree, datasetDict):
    plt.figure(2)
    x = range(1,150,3)
    times = 3 # 设种情况测量times次
    Query = generateQuery(10,tree,datasetDict)
    # 首先是暴力
    y = []
    for k in x:
        s = time()
        for i in range(times):
            forceIKNN(Query, k, tra2point, datasetDict)
        e = time()
        y.append((e-s)/times)
    
    plt.plot(x, y, color="black", label="Force Search")
    y = []
    for k in x:
        s = time()
        for i in range(times):
            IKNN(Query, k, point2tra, tra2point,tree, datasetDict)
        e = time()
        y.append((e-s)/times)
    
    plt.plot(x, y, color="red", label="IKNN")
    plt.xlabel("Search Trajectories' Number")
    plt.ylabel("Average Time")

def main():
    if not os.path.exists(preprocess.dataSetPath) and not os.path.exists(preprocess.newDataSetPath):
        print("Error:","dataSetPath","not exist")
        return

    if not os.path.exists(preprocess.newDataSetPath):
        preprocess.writeDataSetToFile()

    if not os.path.exists(outfilep2tra) or not os.path.exists(outfiletra2p):
        save()    
    tra2point, point2tra, tree, datasetDict = load()
    testIKNNResultIsRight(tra2point, point2tra, tree, datasetDict)
    # 一个样例的可视化
    showOneExample(tra2point, point2tra, tree, datasetDict)
    # 检测效率 时间过长默认注释
    # testDifferentIKNN(tra2point, point2tra, tree, datasetDict)
    plt.show()


if __name__ == "__main__":
    main()