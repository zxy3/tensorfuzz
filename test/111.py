'''
    子数组问题
'''
def sub_array(array, num):
    i = 0
    j = 0
    qmin = []
    qmax = []
    result = 0
    while i < len(array):
        while j < len(array):
            while qmin and array[qmin[-1]] >= array[j]:
                qmin.pop()
            qmin.append(j)
            while qmax and array[qmax[-1]] <= array[j]:
                qmax.pop()
            qmax.append(j)
            if array[qmax[0]] - array[qmin[0]] > num:
                break
            j += 1
        if qmin[0] == i:
            qmin.pop(0)
        if qmax[0] == i:
            qmax.pop(0)
        result += j - i
        i += 1
    return result
# re = sub_array([3, 6, 4, 3, 2], 2)
# print(re)

'''
    汉诺塔问题
'''


def process(num, left, mid, right, from_, to_):
    if num == 1:
        if from_ == mid or to_ == mid:
            print("Move 1 from", from_, " to ", to_)
            return 1
        else:
            print("Move 1 from ", from_, " to ", mid)
            print("Move 1 from ", mid, " to ", to_)
            return 2

    if from_ == mid or to_ == mid:
        another = right if (to_ == left or from_ == left) else left
        part1 = process(num - 1, left, mid, right, from_, another)
        print("Move ", num, " from ", from_, " to ", to_)
        part2 = 1
        part3 = process(num - 1, left, mid, right, another, to_)
        return part1 + part2 + part3
    else:
        part1 = process(num - 1, left, mid, right, from_, to_)
        print("Move ", num, " from ", left, " to ", mid)
        part2 = 1
        part3 = process(num - 1, left, mid, right, to_, from_)
        print("Move ", num, " from ", mid, " to ", to_)
        part4 = 1
        part5 = process(num - 1, left, mid, right, from_, to_)
        return part1 + part2 + part3 + part4 + part5


def hanni(num, left, mid, right):
    return process(num, left, mid, right, left, right)


# re = hanni(2, "left", "mid", "right")
# print(re)

'''
    数组与窗口问题
'''


# def max_sum(array, window):
#     qmax = []
#     i = 0
#     sum = 0
#     while i + window <= len(array):
#         j = i
#         while j < i + 3:
#             while qmax and array[qmax[-1]] <= array[j]:
#                 qmax.pop()
#             qmax.append(j)
#             j += 1
#         sum += array[qmax[0]]
#         if qmax[0] == i:
#             qmax.pop(0)
#         i += 1
#     return sum
#
#
# print(max_sum([4, 2, 4, 5, 6, 3, 6], 3))
# input_array = []
# array_input = input().split(' ')
# for item in array_input:
#     input_array.append(int(item))
#
# window_size_input = input()
# window_size = int(window_size_input)
#
# def max_sum(array, window):
#     qmax = []
#     i = 0
#     sum = 0
#     while i + window <= len(array):
#         j = i
#         while j < i + 3:
#             while qmax and array[qmax[-1]] <= array[j]:
#                 qmax.pop()
#             qmax.append(j)
#             j += 1
#         sum += array[qmax[0]]
#         if qmax[0] == i:
#             qmax.pop(0)
#         i += 1
#     return sum
# print(max_sum(input_array, window_size))

# matrix = [[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]]
# sub_matrix = matrix[1][1 : 1 + 3]
# print(sub_matrix)
# if all(sub_matrix) == 1:
#     print("true")

array1 = [100, 99, 98, 1, 2, 3]
array2 = [1, 2, 3, 4, 5, 40]

# def swap(arr1, arr2, i, j):
#     flag = False
#     difference_before_swap = abs(sum(arr1) - sum(arr2))
#     difference_after_swap = abs(sum(arr1) - arr1[i] + arr2[j] - (sum(arr2) - arr2[j] + arr1[i]))
#     if difference_after_swap < difference_before_swap:
#         flag = True
#         temp1 = arr1[i]
#         temp2 = arr2[j]
#         arr1[i] = arr2[j]
#         arr2[j] = arr1[i]
#     return flag, arr1, arr2
#
#
# flag = False
# i = 0
# while i < len(array1):
#     j = 0
#     while j < len(array2):
#         flag, arr1, arr2 = swap(array1, array2, i, j)
#         if flag is True:
#             array1 = arr1
#             array2 = arr2
#             i = 0
#             break
#         else:
#             j += 1
#     if flag is True:
#         i = 0
#     else:
#         i += 1


# print(abs(sum(array1) - sum(array2)))

# def twoSum(nums, target):
#     """
#     :type nums: List[int]
#     :type target: int
#     :rtype: List[int]
#     """
#     result = []
#     i = 0
#     while i < len(nums) - 1:
#         j = i + 1
#         while j < len(nums):
#             if nums[i] + nums[j] == target:
#                 result.append(i)
#                 result.append(j)
#             j += 1
#         i += 1
#
#     return result
# print(twoSum([3, 2, 4], 6))
#
# str = '123'
# re = list(str)
# print(re)
# class Solution(object):
#     def addTwoNumbers(self, l1, l2):
#         """
#         :type l1: ListNode
#         :type l2: ListNode
#         :rtype: ListNode
#         """
#         str1, str2 = '', ''
#         if l1:
#             str1 += str(l1.val)
#             l1_ = l1.next
#             while l1_:
#                 str1 += str(l1_.val)
#                 l1_ = l1_.next
#         if l2:
#             str2 += str(l2.val)
#             l2_ = l2.next
#             while l2_:
#                 str2 += str(l2_.val)
#                 l2_ = l2_.next
#
#         a1 = list(str1).reverse()
#         a2 = list(str2).reverse()
#         str1, str2 = '', ''
#         for item in a1:
#             str1 += item
#         for item in a2:
#             str2 += item
#         sum = list(str(int(str1) + int(str2))).reverse()

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        str1, str2 = '', ''
        if l1:
            str1 += str(l1.val)
            l1_ = l1.next
            while l1_:
                str1 += str(l1_.val)
                l1_ = l1_.next
        if l2:
            str2 += str(l2.val)
            l2_ = l2.next
            while l2_:
                str2 += str(l2_.val)
                l2_ = l2_.next

        a1 = list(str1).reverse()
        a2 = list(str2).reverse()
        str1, str2 = '', ''
        for item in a1:
            str1 += item
        for item in a2:
            str2 += item
        sum = list(str(int(str1) + int(str2))).reverse()
        result = []
        for item in sum:
            result.append(int(item))
        return result