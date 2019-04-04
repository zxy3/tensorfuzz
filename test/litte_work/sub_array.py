

input_array = input().split(' ')
array = []
for item in input_array:
    array.append(int(item))

num = int(input())

qmin = []
qmax = []
i = 0
j = 0
result = 0
# all_sub_array = []
# while i < len(array):
#     while j < len(array):
#         while qmin and array[qmin[-1]] < array[j]:
#             qmin.pop()
#         qmin.append(j)
#         while qmax and array[qmax[-1]] <= array[j]:
#             qmax.pop()
#         qmax.append(j)
#         if array[qmax[0]] - array[qmin[0]] > num:
#             break
#         j += 1
#     if qmin[0] == i:
#         qmin.pop(0)
#     if qmax[0] == i:
#         qmax.pop(0)
#     j1 = i + 1
#     while j1 <= j:
#         sub_array = array[i: j1]
#         if sub_array not in all_sub_array:
#             all_sub_array.append(sub_array)
#             result += 1
#         j1 += 1
#     i += 1
# print(result)

i = 0
while i < len(array) - 1:
    j = i + 1
    while j < len(array):
        if j + 1 == len(array):
            sub_array = array[i:]
        else:
            sub_array = array[i: j + 1]
        if max(sub_array) - min(sub_array) > num:
            result += 1
        j += 1
    i += 1
print(result)
