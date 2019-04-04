'''
输入一个数组和一个数字，在数组中查找两个数，使得它们的和正好是输入的那个数字，统计这样两个数的对数。
输入第一行是数组，每一个数用空格隔开；第二行是数字和。
'''
array_input = input()
sum_input = input()
array = [int(item) for item in array_input.split(' ')]
sum = int(sum_input)

array.sort()
num = 0
i = 0
while i + 1 < len(array):
    j = i + 1
    if array[i] > sum:
        break
    while j < len(array):
        if array[j] > sum:
            break
        if array[i] + array[j] == sum:
            num += 1
        j += 1
    i += 1
print(num)