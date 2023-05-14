import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


# pc_normalize为point cloud normalize
# 即将点云数据进行归一化处理
# 归一化点云，使用已centroid为中心的坐标，球半径为1
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)             # 压缩点云数据求得x,y,z的均值
    pc = pc - centroid                         # 求得每一点到中点的绝对距离
    m = np.max(np.sqrt(np.sum(pc**2, axis=1))) # 求得离中心点最大距离，最大的标准差
    pc = pc / m                                # 归一化，这里使用的是Z-score标准化方法，即为(x-mean)/std
    return pc


# 确定每个点到采样点的距离，用于ball_query过程
# 欧式距离
# 函数输入是两组点，N为第一组点src个数，M为第二组点dst个数，C为输入点的通道数（如果xyz时C=3）
# 函数返回的是两组点两两之间的欧式距离，即N*M矩阵
# 函数训练中数据以Mini-Batch的形式输入，所以一个Batch数量的维度为B
def square_distance(src, dst):
    # 由于在训练中数据通常是以Mini-Batch的形式输入的
    # 所以有一个Batch数量的维度为B。
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # matmul矩阵相乘，2*(xn * xm + yn * ym + zn * zm)
    # 为了保证src和dst矩阵可以相乘，这里涉及到三维矩阵乘法
    # 需要将dst转变一下维度[B, C, M]
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # xn*xn + yn*yn + zn*zn
    dist += torch.sum(src**2, -1).view(B, N, 1)
    # xm*xm + ym*ym + zm*zm
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


# 按照输入的点云数据和索引返回索引的点云数据
# 例如Points为B*2048*3点云，idx为[5.666，1000.2000]
# 则返回Batch中第5666，1000，2000个点组成的B*4*3的点云集
# 如果idx为一个[B,D1,''''DN],则它会按照idx中的纬度结构将其提取成[B，D1，‘’‘DN,C]
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]#   #点云数据points
        idx: sample index data, [B, S]   #点云索引idx
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # idx为每各样本中所要选取的点的索引
    # 这里输入的点云数据B*2048*3，其中B为Batch_size，样本数
    # 简单来说，这个函数就是要再点云数据中选取每个样本中索引值在idx这个索引数组里面的点
    # idx的长度为4时，则最后输出的为B*4*3，也就是在2048个点中选取在索引值idx的点
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    # view_shape=[B,1]
    view_shape[1 :] = [1] * (len(view_shape) - 1)
    # repeat_shape=[B,S]
    repeat_shape = list(idx.shape)
    # repeat_shape=[B,S]
    repeat_shape[0] = 1
    # .view(view_shape)=.view(B,1)
    # .repeat(repeat_shape)=.view(1,S)
    # 综上所述，batch_indices的维度[B,S]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # 从points当中取出每个batch_indices对应索引的数据点
    new_points = points[batch_indices, idx, :]
    return new_points


# 最远点采样
# farthest_point_sample函数完成最远点采样
# 从一个输入点云中按照所需要的点的个数npoint采样出足够多的点
# 并且点与点的距离需要足够远
# 返回结果是npoint个采样点再原始点云中的索引
def farthest_point_sample(xyz, npoint):
    """
    FPS farthest point sample 最远点采样

    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)    # [B, npoint] 初始化一个 centrdis 矩阵 ，用于存储 npoint 个采样点的索引位置
    distance = torch.ones(B, N).to(device) * 1e10                      # [B, N] distance 矩阵，记录batch中所有点到某一个点的距离，初始化值很大，后面会迭代更新
                                                                       # [B] \in[0,N]: farthest 表示当前最远的点, 也是随机初始化, 范围为0-N, 初始化B个; 每个batch都随机有一个初始化最远点  # 记录某个样本中所有点到某一个点的距离
    farthest = torch.randint(0, N, (B, ), dtype=torch.long).to(device) # batch里每个样本随机初始化一个最远点的索引
                                                                       
                                                                       
    batch_indices = torch.arange(B, dtype=torch.long).to(device)# batch_indices初始化为0-(B-1)的数组

    for i in range(npoint):                                            # 更新第i个最远点
                                                                       # 假设当前采样点centroids为当前的最远点farthest
        centroids[:, i] = farthest                                     #第一个采样点选随机初始化的索引
                                                                       # 取出该中心点centroid的坐标 # 取出这个最远点的xyz坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)       #得到当前采样点的坐标 B*3  # 计算点集中的所有点到这个最远点的欧式距离
                                                                       # 取出该中心点centroid点的欧式距离,存到dist矩阵中
        dist = torch.sum((xyz - centroid)**2, -1)                      #计算当前采样点与其他点的距离    # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
                                                                       # 建立一个mask,如果dist中的元素小于distance矩阵中保存的距离值,则更新distance中的对应值
                                                                       # 随着迭代的继续,distance矩阵中的值会慢慢变小
                                                                       # 其相当于记录着某个batch中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance                                         #选择距离最近的来更新距离（更新维护这个表）
        distance[mask] = dist[mask]                                    #从distance矩阵中去除最远的点为farthest,继续下一轮迭代
        farthest = torch.max(distance, -1)[1]                          #重新计算得到最远点索引（在更新的表中选择距离最大的那个点）
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    寻找球形领域中的点
    Input:
        - `radius`: local region radius 输入中radius为球形领域的半径
        - `nsample`: max sample number in local region 每个领域中要采样的点
        - `xyz`: all points, [B, N, 3] 所有的点云
        - `new_xyz`: query points, [B, S, 3] S个球形领域的中心（由最远点采样在前面得出）
    Return:
        - `group_idx`: grouped points index, [B, S, nsample] 每个样本的每个球形领域的nsample个采样点集的索引
    """

    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(
        N,
        dtype=torch.long,
    ).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists:[B,S,N]记录S个中心点（new_xyz）与所有点（xyz）之间的欧氏距离
    sqrdists = square_distance(new_xyz, xyz)                               #得到B N M （就是N个点中每一个和M中每一个的欧氏距离）   # sqrdists: [B, S, N] 记录中心点与所有点之间的欧几里德距离
    group_idx[sqrdists > radius**2
             ] = N                                                         #找到距离大于给定半径的设置成一个N值（1024）索引 # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    group_idx = group_idx.sort(
        dim=-1
    )[0][:, :, : nsample]                                                  #做升序排序，后面的都是大的值（1024）# 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
                                                                           # group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample]) #如果半径内的点没那么多，就直接用第一个点来代替了。。。
                                                                           # 找到group_idx中值等于N的点
    mask = group_idx == N
                                                                           # 将这些点的值替换为第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx


# Sampling+Grouping主要用于将整个点云分散成局部的group
# 对于每一个group都可以用PointNet单独的提取局部的全局特征
# Sampling+Grouping分成了sampl_and_group和sampl_and_group_all两个函数
# 其区别在于sample_and_group_all直接将所有点作为一个group
# 例如：
# 512=npoint:poins sampled in farthest point sampling
# 0.2=radius:search radius in local region
# 32=nsample:how many points in each local region
# 将整个点云分散成局部的group，对每一个group都可以用PointNet单独的提取局部的全局特征
def sample_and_group(
    num_group: int,
    radius: float,
    nsample: int,
    xyz: int,
    points: torch.Tensor,
    returnfps: bool = False,
):
    """
    Input:
        num_group:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, num_group, nsample, 3]
        new_points: sampled points data, [B, num_group, nsample, 3+D]
    """
    B, N, C = xyz.shape
    # 从原点云通过最远点采样挑出的采样点作为new_xyz：
    # 先用farhest_point_sample函数实现最远点采样得到采样点的索引
    # 再通过index_points将这些点的从原始点中挑出来，作为new_xyz
    fps_idx = farthest_point_sample(xyz, num_group) # [B, num_group, C]
    torch.cuda.empty_cache()

    # 从原点云中挑出最远点采样的采样点为 new_xyz
    new_xyz = index_points(xyz, fps_idx) # 中心点 [B, S, 3]
    torch.cuda.empty_cache()

    # idx:[B,num_group,nsample]，代表 num_group 个球形区域中每个区域的nsample个采样点的索引
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()             # idx:[B, num_group, nsample] 代表 num_group 个球形区域中每个区域的nsample个采样点的索引
                                         # grouped_xyz:[B, num_group, nsample, C]
                                         # 通过index_points将所有group内的nsample个采样点从原始点中挑出来
    grouped_xyz = index_points(xyz, idx) # [B, num_group, nsample, C]
    torch.cuda.empty_cache()

    # grouped_xyz减去采样点即中心值
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, num_group, 1, C)
    torch.cuda.empty_cache()

    # 如果每个点上有新的特征维度，则拼接新的特征与旧的特征，否则直接返回旧的特征
    # 注：用于拼接点的特征数据和点坐标数据
    # 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    if points is not None:
        # 通过index_points将所有group内的nsample个采样点从原始点中挑出来，得到group内点的除坐标维度外的其他维度的数据
        grouped_points = index_points(points, idx)
        # dim=-1代表按照最后的维度进行拼接，即相当于dim=3
        new_points = torch.cat([
            grouped_xyz_norm,
            grouped_points,
        ], dim=-1)               # [B, num_group, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

