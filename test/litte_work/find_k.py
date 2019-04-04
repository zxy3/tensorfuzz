'''
    找到给定数组的给定区间内的倒数第K小的数值
    输入的第一行为数组，每一个数用空格隔开；第二行是区间（第几个数到第几个数，两头均包含），
    两个值用空格隔开；第三行为K值。
'''
array_input = input()
scope_input = input()
k_input = input()

array = [int(item) for item in array_input.split(' ')]
scope = [int(item) for item in scope_input.split(' ')]
k = int(k_input)

sub_array = array[scope[0] - 1: scope[1]]
sub_array.sort()
print(sub_array[len(sub_array) - k])


