'''
有两个序列 a,b，大小都为 n,序列元素的值任意整数，无序； 要求：通过交换 a,b 中的元素，使[序列 a 元素的和]与[序列 b 元素的和]之间的差最小。
输入为两行，分别为两个数组，每个值用空格隔开。
'''
import copy
array1_input = input()
array2_input = input()
array1 = [int(item) for item in array1_input.split(' ')]
array2 = [int(item) for item in array2_input.split(' ')]

def swap(arr1, arr2, i, j):
    flag = False
    difference_before_swap = abs(sum(arr1) - sum(arr2))
    difference_after_swap = abs(sum(arr1) - arr1[i] + arr2[j] - (sum(arr2) - arr2[j] + arr1[i]))
    if difference_after_swap < difference_before_swap:
        flag = True
        temp1 = arr1[i]
        temp2 = arr2[j]
        arr1[i] = arr2[j]
        arr2[j] = arr1[i]
    return flag, arr1, arr2


flag = False
i = 0
while i < len(array1):
    j = 0
    while j < len(array2):
        flag, arr1, arr2 = swap(array1, array2, i, j)
        if flag is True:
            array1 = arr1
            array2 = arr2
            i = 0
            break
        else:
            j += 1
    if flag is True:
        i = 0
    else:
        i += 1


print(abs(sum(array1) - sum(array2)))