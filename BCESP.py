import pickle
import sys
import random
from collections import defaultdict
from math import acos, inf
import subprocess
from time import perf_counter

import numpy as np

from d import DataReorganizer

# 常量定义
inf = float('inf')  # 无穷大，用于比较
pi = acos(-1.0)  # π值

# 图类，用于存储分区算法所需的数据结构和方法
class GraphPartitioner2:
    def __init__(self, maxvertex, numberOfPartition,data_path):
        self.totaledge = 0
        self.data_path = data_path
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
        self.uidW, self.iidW = self.load_embeddings()  # 加载预训练的嵌入向量
        self.count = 0 #用于记录已分配边的数量
    def load_embeddings(self):
        # 加载预训练的嵌入向量
        with open(self.data_path + '/user_pretrain.pk', 'rb') as f:
            uidW = pickle.load(f)
        with open(self.data_path + '/item_pretrain.pk', 'rb') as f:
            iidW = pickle.load(f)
        return uidW, iidW

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
    # 计算两个嵌入向量之间的余弦相似度
    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        return dot_product / (norm_vec1 * norm_vec2)

    # 计算边与分区的相似性
    def getEdgePartitionSimilarity(self, src, dest, partition):
        src_embedding = self.uidW[src]
        dest_embedding = self.iidW[dest-self.max_src_value-1]
        # 获取属于特定分区的所有顶点的嵌入向量
        partition_embeddings = []
        for v in self.partitionOfVertices:
            if partition in self.partitionOfVertices[v]:
                vertex_embedding = self.uidW[v] if v <= self.max_src_value else self.iidW[v - self.max_src_value - 1]
                partition_embeddings.append(vertex_embedding)

        if not partition_embeddings:
            return 0
        avg_embedding = np.mean(partition_embeddings, axis=0)
        similarity_src = self.cosine_similarity(src_embedding, avg_embedding)
        similarity_dest = self.cosine_similarity(dest_embedding, avg_embedding)
        return (similarity_src + similarity_dest) / 2

    # 修改getPartitionNumberOfEdge方法以包含rat值，并使用原始dest值进行分区
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
            if self.count <= self.totaledge * 0.001:
                similarity = self.getEdgePartitionSimilarity(src, dest, partition)
                totalScore = scoreReplicationFactor + scoreBalance + similarity  # 加入相似性分数
            else:
                totalScore = scoreReplicationFactor + scoreBalance
            maxscore = totalScore
            partitionId = partition
            if totalScore < scoreHDRF:
                if totalScore > maxscore:
                    maxscore = totalScore
                    partitionId = partition
                continue
            elif totalScore > scoreHDRF:
                scoreHDRF = totalScore
                candidateList.clear()
                candidateList.append(partition)
            else:
                candidateList.append(partition)
        if candidateList :
            partitionId = random.choice(candidateList)

        self.addToPartition(src, partitionId)
        self.addToPartition(dest, partitionId)
        self.addEdge(src, dest, rat, partitionId)  # 使用原始dest值添加边
        return partitionId  # 保存划分结果为边的形式

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
        data_reorganizer = DataReorganizer('.../RecUL/Data/Process/BookCrossing/0.02/train.csv', 'BookCrossing_reorganized0.02.txt')
        self.max_src_value, self.totaledge = data_reorganizer.run()
        print("totaledge:",self.totaledge)
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
                self.count += 1
                self.getPartitionNumberOfEdge(src, dest, rat)


        end = perf_counter()
        print("Time taken:", (end - begin), file=sys.stderr)
        self.savePartitionResultsAsEdges(output_filename)
        self.printEdgeCountPerPartition()

    def get_partition_results(self):
        # print(self.C)
        return self.C, self.C_itr



