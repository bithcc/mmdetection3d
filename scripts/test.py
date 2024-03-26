import torch

# 假设A和B是给定的tensor
A = torch.tensor([[1,5,1],[2,1,2],[3,6,3],[4,3,4],[5,2,5],[6,4,6]])  # 体素的三个坐标
B = torch.tensor([[10,10,10],[20,20,20],[30,30,30],[40,40,40],[50,50,50],[60,60,60]])  # 对应体素的特征

# 按照A的第二列正排序
_, indices_pos = torch.sort(A[:, 1], descending=False)
C = A[indices_pos]
E = B[indices_pos]

# 按照A的第二列逆排序
_, indices_neg = torch.sort(A[:, 1], descending=True)
D = A[indices_neg]
F = B[indices_neg]

# E和F按行方向相加
sum_EF = E + F

# # 将结果重新变换到A的排序方式，由于正排序的逆操作即为逆排序，反之亦然
# # 我们可以通过对排序索引再次进行排序获取原始顺序的索引映射
# _, indices_to_original = torch.sort(indices_pos)

# # 应用索引映射，恢复到原始A的排序
# result = sum_EF[indices_to_original]

# print('C:',C)
# print('E:',E)

# print('D:',D)
# print('F:',F)

# print('sum_EF:',sum_EF)
# print('result:',result)
print('E',E)
print('F',F)

print('B',B)
num_rows_to_replace=sum_EF.shape[0]//3
indices_to_replace = indices_pos[:num_rows_to_replace]
print('index',indices_pos)
B[indices_to_replace]=sum_EF[:num_rows_to_replace]
print('sum_EF',sum_EF)
print('B',B)

