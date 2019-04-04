def reverse_sum(array):
    sum = 0
    # i = 0
    # while i < len(array) - 1:
    #     j = i + 1
    #     while j < len(array):
    #         if array[i] > array[j]:
    #             sum += 1
    #         j += 1
    #     i += 1
    temp_array = []
    for item in array:
        temp_array.append(item)
    array.sort()
    dict1, dict2 = {}, {}
    for i in range(len(array)):
        dict1[array[i]] = i
        dict2[temp_array[i]] = i

    for i in range(len(array)):
        index = dict2[array[i]]
        temp = abs((len(array) - i) - (len(array) - index))
        print(temp)
        sum += abs((len(array) - i) - (len(array) - index))

    print(sum)

num_test_cases = int(input())
for i in range(num_test_cases):
    size_of_array = int(input())
    array_input = input().split(' ')
    array = [int(item) for item in array_input]
    reverse_sum(array)