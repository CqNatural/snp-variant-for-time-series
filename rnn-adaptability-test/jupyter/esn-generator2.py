#coding=utf-8

import random

matrix = []
matrix_ext = []

poch = 15

# 输出3*(5)到17*(5)
for i in range(poch):
    n = i + 3
    a = []
    b = []
    for j in range(n):
        a.append(5)
        b.append(5)
    a.append(3)
    a.append(4)
    a.append(-8)
    b.append(3)
    b.append(3)
    b.append(-5)
    for j in range(n-1):
        a.append(-5)
        b.append(-5)
    a.append(-4)
    b.append(-6)
    matrix.append(a)
    matrix.append(b)
# 缺少18*(5)和19*(5)
for i in range(2):
    n = i + 40  # 这里就用18
    a = []
    b = []
    for j in range(n):
        a.append(5)
        b.append(5)
    a.append(3)
    a.append(4)
    a.append(-8)
    b.append(3)
    b.append(3)
    b.append(-5)
    for j in range(n-1):
        a.append(-5)
        b.append(-5)
    a.append(-4)
    b.append(-6)
    matrix_ext.append(a)
    matrix_ext.append(b)    
# 输出20*(5)到34*(5)
for i in range(poch):
    n = i + 3+ poch + 2
    a = []
    b = []
    for j in range(n):
        a.append(5)
        b.append(5)
    a.append(3)
    a.append(4)
    a.append(-8)
    b.append(3)
    b.append(3)
    b.append(-5)
    for j in range(n-1):
        a.append(-5)
        b.append(-5)
    a.append(-4)
    b.append(-6)
    matrix.append(a)
    matrix.append(b)

sum = 0
# print(matrix)
for i in range(1200):
    vec = matrix[random.randint(0,poch*2)]
    for j in range(len(vec)):
        element = vec[j]
        sum += element
        # print(f"{element}.0e-00")
        print(f"{sum}.0e-00")

print("------------")
# print(matrix_ext)
for i in range(4):
    vec = matrix_ext[random.randint(0,3)]
    for j in range(len(vec)):
        element = vec[j]
        sum += element
        # print(f"{element}.0e-00")
        print(f"{sum}.0e-00")
