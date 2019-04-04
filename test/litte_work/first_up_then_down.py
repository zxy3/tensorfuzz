'''
    从一列数中筛除尽可能少的数使得从左往右看，这些数是从小到大再从大到小的。
    输入时一个数组，数值通过空格隔开。
    输出筛选之后的数组，用空格隔开。如果有多种解雇哦，则一行一种结果。
'''
array_input = input()
array = [int(item) for item in array_input]
