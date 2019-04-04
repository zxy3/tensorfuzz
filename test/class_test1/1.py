def mySort(array):
    array.sort()
    dict = {}
    for item in array:
        if item not in dict.keys():
            dict[item] = 1
        else:
            dict[item] += 1
    dict = sorted(dict.items(), key=lambda d: d[1], reverse=True)
    for j in range(len(dict)):
        item = dict[j]
        for i in range(item[1]):
            if j == len(dict) - 1 and i == item[1] - 1:
                print(item[0], end="")
            else:
                print(item[0], end=" ")

# result = mySort([5, 5, 4, 6, 4])
# result = mySort([5,1, 3, 5,3,1, 19,5, 6, 3,2, 3,2, 5, 3,2, 5,3, 1])
# print(result)

num_test_cases = int(input())
for i in range(num_test_cases):
    size_of_array = int(input())
    array_input = input().split(' ')
    array = [int(item) for item in array_input]
    mySort(array)