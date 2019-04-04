def relative_sort(array1, array2):
    array1.sort()
    result = []
    for item in array2:
        for item1 in array1:
            if item1 == item:
                result.append(item1)

        while item in array1:
            array1.remove(item)

    result = result + array1
    for i in range(len(result)):
        if i == len(result) - 1:
            print(result[i], end='')
        else:
            print(result[i], end=' ')

# relative_sort([2, 1, 2, 5, 7, 1, 9, 3, 6, 8, 8], [2, 1, 8, 3])
num_test_cases = int(input())
for i in range(num_test_cases):
    size_of_arrays_input = input().split(' ')
    array1_size = int(size_of_arrays_input[0])
    array2_size = int(size_of_arrays_input[1])
    array1_input = input().split(' ')
    array1 = [int(item) for item in array1_input]
    array2_input = input().split(' ')
    array2 = [int(item) for item in array2_input]
    relative_sort(array1, array2)
