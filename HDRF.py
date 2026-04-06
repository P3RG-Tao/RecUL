import sys
import random
from collections import defaultdict
from math import acos, inf
import subprocess
from time import perf_counter

from d import DataReorganizer
from d5 import DataReorganizer2

# 常量定义
inf = float('inf')  # 无穷大，用于比较
pi = acos(-1.0)  # π值

# 图类，用于存储分区算法所需的数据结构和方法
class GraphPartitioner:
    def __init__(self, maxvertex, numberOfPartition):
        self.maxvertex = max(int(maxvertex), 1)  # 确保最大顶点数为整数
        self.numberOfPartition = numberOfPartition  # 分区数量
        self.partitionSize = [0] * self.maxvertex  # 每个分区的大小
        self.partitionOfVertices = defaultdict(set)  # 每个顶点属于哪些分区
        self.degree = [0] * self.maxvertex  # 每个顶点的度数
        self.numberOfEdges = [0] * numberOfPartition  # 每个分区的边数
        self.maxLoad = 0  # 最大负载
        self.edgesInPartition = [[] for _ in range(numberOfPartition)]  # 每个分区的边列表
        self.C = [{} for _ in range(numberOfPartition)]  # 初始化C列表
        self.C_itr = [[] for _ in range(numberOfPartition)]  # 初始化C_itr列表
        self.max_src_value = 0  # 存储第一列的最大值

    # 获取最小分区的索引
    def getIndexOfSmallestPartition(self, s):
        MAX_PARTITION_SIZE = 1e18
        minSize = MAX_PARTITION_SIZE
        idxMin = -1
        candidateList = []
        if not s:
            for i in range(self.numberOfPartition):
                if self.partitionSize[i] < minSize:
                    minSize = self.partitionSize[i]
                    idxMin = i
                    candidateList.clear()
                    candidateList.append(i)
                elif self.partitionSize[i] == minSize:
                    candidateList.append(i)
        else:
            for partitionIdx in s:
                if self.partitionSize[partitionIdx] < minSize:
                    minSize = self.partitionSize[partitionIdx]
                    idxMin = partitionIdx
                    candidateList.clear()
                    candidateList.append(partitionIdx)
                elif self.partitionSize[partitionIdx] == minSize:
                    candidateList.append(partitionIdx)
        return random.choice(candidateList)

    # 获取两个集合的交集
    def getIntersection(self, s1, s2):
        return s1.intersection(s2)

    # 检查顶点是否存在于某个分区
    def isVertexExistInPartition(self, partition, vertex):
        return partition in self.partitionOfVertices[vertex]

    # 获取最小分区的大小
    def getMinSize(self):
        return min(self.numberOfEdges)

    # 增加分区的负载
    def incrementMachineLoad(self, partitionId):
        self.numberOfEdges[partitionId] += 1
        self.maxLoad = max(self.maxLoad, self.numberOfEdges[partitionId])

    # 增加分区的大小
    def addPartitionSize(self, idxPartition):
        self.partitionSize[idxPartition] += 1

    # 将顶点添加到分区
    def addToPartition(self, vertex, idxPartition):
        self.partitionOfVertices[vertex].add(idxPartition)

    # 添加边并更新分区负载
    # 添加边并更新分区负载
    # 修改addEdge方法以保存调整后的dest值
    def addEdge(self, src, dest, rat, idxPartition):
        adjusted_dest = dest - self.max_src_value - 1  # 调整dest值
        self.degree[src] += 1
        self.degree[adjusted_dest] += 1
        self.incrementMachineLoad(idxPartition)

        # 更新C列表
        if src not in self.C[idxPartition]:
            self.C[idxPartition][src] = []
        self.C[idxPartition][src].append(adjusted_dest)

        # 更新C_itr列表
        self.C_itr[idxPartition].append([src, adjusted_dest, rat])

        # 保存边到分区，包括rat值
        self.edgesInPartition[idxPartition].append((src, adjusted_dest, rat))

    # 计算顶点的theta值
    def getTheta(self, src, dest):
        degreeSrc = self.degree[src] + 1
        degreeDest = self.degree[dest] + 1
        sumDegree = degreeSrc + degreeDest
        thetaSrc = degreeSrc / sumDegree
        thetaDest = degreeDest / sumDegree
        return thetaSrc, thetaDest

    # 修改getPartitionNumberOfEdge方法以包含rat值，并使用原始dest值进行分区
    def getPartitionNumberOfEdge(self, src, dest, rat):

        if src >= self.maxvertex or dest >= self.maxvertex:
            # 如果顶点编号超出范围，扩展degree列表
            self._extendDegreeList(max(src, dest))

        candidateList = []
        scoreHDRF = 0
        partitionId = -1

        for partition in range(self.numberOfPartition):
            thetaSrc, thetaDest = self.getTheta(src, dest)
            gSrc = 2.0 - thetaSrc if self.isVertexExistInPartition(partition, src) else 0
            gDest = 2.0 - thetaDest if self.isVertexExistInPartition(partition, dest) else 0

            scoreReplicationFactor = gSrc + gDest
            scoreBalance = (self.maxLoad - self.numberOfEdges[partition]) / (1 + self.maxLoad - self.getMinSize())
            totalScore = scoreReplicationFactor + scoreBalance
            if totalScore < scoreHDRF:
                continue
            elif totalScore > scoreHDRF:
                scoreHDRF = totalScore
                candidateList.clear()
                candidateList.append(partition)
            else:
                candidateList.append(partition)

        partitionId = random.choice(candidateList)

        self.addToPartition(src, partitionId)
        self.addToPartition(dest, partitionId)
        self.addEdge(src, dest, rat, partitionId)  # 使用原始dest值添加边
        return partitionId    # 保存划分结果为边的形式
    # 保存划分结果为边的形式，并区分各分区
    # 修改保存划分结果为边的形式到文件的方法，包含rat值
    def savePartitionResultsAsEdges(self, filename):
        with open(filename, 'w') as f:
            for partition in range(self.numberOfPartition):
                f.write(f"Partition {partition + 1}:\n")
                for edge in self.edgesInPartition[partition]:
                    f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
                f.write("\n")

    # 新增方法：打印每个分区的边数
    def printEdgeCountPerPartition(self):
        print("Partition Edge Counts:")
        for i in range(self.numberOfPartition):
            print(f"Partition {i+1}: {len(self.edgesInPartition[i])} edges")

    def _extendDegreeList(self, new_max):
        new_size = max(self.maxvertex, new_max + 1)
        self.degree.extend([0] * (new_size - len(self.degree)))
        self.partitionSize.extend([0] * (new_size - len(self.partitionSize)))
        self.maxvertex = new_size

    def run_partitioning(self, input_filename, output_filename):
        data_reorganizer = DataReorganizer2('E:/wu/IFRU-main -gai/Data/Process/BookCrossing/0.02/train.csv', 'Mooccube_reorganized0.01.txt')
        self.max_src_value = data_reorganizer.run()
        begin = perf_counter()

        # 读取处理好的文件
        with open(input_filename, 'r') as f:
            for line_number, line in enumerate(f, 1):
                parts = line.split()
                if len(parts) != 3 or not all(part.isdigit() for part in parts):
                    continue
                src, dest, rat = map(int, parts)
                if src > dest:
                    src, dest = dest, src
                self.getPartitionNumberOfEdge(src, dest, rat)

        end = perf_counter()
        print("Time taken:", (end - begin), file=sys.stderr)
        self.savePartitionResultsAsEdges(output_filename)
        self.printEdgeCountPerPartition()

    def get_partition_results(self):
        # print(self.C)
        return self.C, self.C_itr



