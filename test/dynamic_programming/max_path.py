import copy

li = [[7], [3, 8], [8, 1, 0], [2, 7, 4, 4], [4, 5, 2, 6, 5]]
max_sum = copy.deepcopy(li)

# print(max_sum)

lowest_level = li[len(li) - 1]
max_sum[len(li) - 1] = copy.deepcopy(lowest_level)

i = len(li) - 2
while i >= 0:
    for j in range(len(li[i])):
        max_sum[i][j] = max(max_sum[i + 1][j], max_sum[i + 1][j + 1]) + li[i][j]
        print('max_sum[', i, '][', j, ']=', max_sum[i][j])

    i -= 1

print(max_sum[0][0])
