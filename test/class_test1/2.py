def compute_min_swap(array):
    dict = {}
    for i in range(len(array)):
        dict[array[i]] = i
    array.sort()
    flag = [False] * len(dict)
    loops = 0
    for i in range(len(array)):
        if flag[i] == False:
            j = i
            while flag[j] == False:
                flag[j] = True
                j = dict[array[i]]
            loops += 1
    print(len(array) - loops)


# print(compute_min_swap([1, 5, 4, 3, 2]))
num_test_cases = int(input())
for i in range(num_test_cases):
    size_of_array = int(input())
    array_input = input().split(' ')
    array = [int(item) for item in array_input]
    compute_min_swap(array)
