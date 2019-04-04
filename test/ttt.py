# input_array = input().split(' ')
# array = [int(item) for item in input_array]
#
# def left_to_right(arr):
#     dp = [1] * len(arr)
#     i = 1
#     while i < len(arr):
#         j = 0
#         while j < i:
#             if arr[j] <= arr[i] and 1 + dp[j] > dp[i]:
#                 dp[i] = 1 + dp[i]
#             j += 1
#         i += 1
#     return dp
#
#
# def right_to_left(arr):
#     dp = [1] * len(arr)
#     i = len(arr) - 2
#     while i >= 0:
#         j = len(arr) - 1
#         while j > i:
#             if arr[j] <= arr[i] and 1 + dp[j] > dp[i]:
#                 dp[i] = 1 + dp[i]
#             j -= 1
#         i -= 1
#     return dp
#
#
# def get_max_indexs(arr, dp_1, dp_2):
#     indexs = [0]
#     for i in range(len(arr)):
#         sum = dp_1[i] + dp_2[i]
#         if sum > dp_1[indexs[0]] + dp_2[indexs[0]]:
#             indexs = [i]
#         elif sum == dp_1[indexs[0]] + dp_2[indexs[0]]:
#             indexs.append(i)
#     return indexs
#
#
# def find_remove_indexes(list1, list2):
#     remove_indexes = []
#     remove_alternative_indexes = []
#
#     for i in range(len(list1)):
#         sub_list = [list1[i], list1[i] + 1]
#         remove_alternative_indexes.append(sub_list)
#     for i in range(len(list2)):
#         sub_list = [list2[i], list2[i] + 1]
#         remove_alternative_indexes.append(sub_list)
#
#     # print(remove_alternative_indexes)
#     help_indexes = []
#     for i in range(pow(2, len(remove_alternative_indexes))):
#         help_index_before_process = str(bin(i))[2:]
#         # print(help_index_before_process)
#         if len(help_index_before_process) != len(remove_alternative_indexes):
#             help_index_before_process = '0' * (len(remove_alternative_indexes) - len(help_index_before_process)) + help_index_before_process
#             help_indexes.append(help_index_before_process)
#         else:
#             help_indexes.append(help_index_before_process)
#     for i in range(len(help_indexes)):
#         help_indexes[i] = list(help_indexes[i])
#         for j in range(len(help_indexes[i])):
#             help_indexes[i][j] = int(help_indexes[i][j])
#     # print(help_indexes)
#     # return remove_alternative_indexes, help_indexs
#     remove_indexes = []
#     if remove_alternative_indexes:
#         for item in help_indexes:
#             sub_indexes = []
#             for i in range(len(item)):
#                 sub_indexes.append(remove_alternative_indexes[i][item[i]])
#             remove_indexes.append(sub_indexes)
#     return remove_indexes
#
# dp1 = left_to_right(array)
# dp2 = right_to_left(array)
# indexs = get_max_indexs(array, dp1, dp2)
# # print(indexs)
#
# # print(dp1)
# # print(dp2)
# # print(indexs)
# # print()
# # results = []
# for index_ in range(len(indexs)):
#     index = indexs[index_]
#     # print('新的情况=======》''index', index, '最大值为', array[index])
#     left_array = array[0: index].copy()
#     right_array = array[index + 1:].copy()
#     # print(left_array)
#     # print(right_array)
#     delta = 0  # index前进的位数
#     for item in left_array:  # 删除大于最终结果最大值的值
#         if item > array[index]:
#             delta += 1
#             left_array.remove(item)
#     for item in right_array:
#         if item > array[index]:
#             right_array.remove(item)
#     temp_array = left_array + [array[index]] + right_array  # 删除大元素后的新数组
#     # print('temp_array:', temp_array)
#     # print('删除不符合要求后的array:', temp_array)
#     dp1_temp = left_to_right(temp_array)
#     dp2_temp = right_to_left(temp_array)
#     # print('新数组的dp1', dp1_temp)
#     # print('新数组的dp2', dp2_temp)
#     # index_temp = get_max_indexs(temp_array, dp1_temp, dp2_temp)[0]
#     index_temp = index - delta  # 新数组最大数的下标
#     # print(index_temp)
#     left_alternative_arrays = []
#     right_alternative_arrays = []
#
#     for i in range(index_temp + 1):
#         if i == 0:
#             left_alternative_arrays.append([[]])
#         else:
#             alternative_nums = []
#             num_increase = dp1_temp[i]  # 以temp_array[i]为最后一位的递增序列的个数
#             for j in range(0, i):
#                 if temp_array[j] <= temp_array[i] and dp1_temp[j] == num_increase - 1:
#                     lis = left_alternative_arrays[j]
#                     for item in lis:
#                         li = item + [temp_array[j]]
#                         alternative_nums.append(li)
#                     # for item in lis:
#                         # li = item + [temp_array[j]]
#                     # li = left_alternative_arrays[j] + [temp_array[j]]
#                     #     alternative_nums.append(li)
#             left_alternative_arrays.append(alternative_nums)
#     left_max_alternative_nums = left_alternative_arrays[-1]
#     # print('左边最长递增子序列候选组：', left_max_alternative_nums)
#
#     # i = len(dp2_temp) - 1
#     # while i >= index_temp:
#     #     if i == len(dp2) - 1:
#     #         right_alternative_arrays.append([[]])
#     #     else:
#     #         alternative_nums = []
#     #         num_decrease = dp2_temp[i]
#     #         j = len(dp2_temp) - 1
#     #         while j > i:
#     #             if temp_array[j] <= temp_array[i] and dp2_temp[j] == num_decrease - 1:
#     #                 lis = right_alternative_arrays[len(dp2_temp) - j - 1]
#     #                 for item in lis:
#     #                     li = item + [temp_array[j]]
#     #                     alternative_nums.append(li)
#     #             j -= 1
#     #         right_alternative_arrays.append(alternative_nums)
#     #     i -= 1
#     # right_max_alternative_nums = right_alternative_arrays[-1]
#     # print('右边最长子序列候选组：', right_max_alternative_nums)
#     reverse_right_array = temp_array[::-1]
#     # print("将右边的数反转", reverse_right_array)
#     dp_temp_reverse = left_to_right(reverse_right_array)
#     index_temp_reverse = len(temp_array) - index_temp - 1
#     for i in range(index_temp_reverse + 1):
#         if i == 0:
#             right_alternative_arrays.append([[]])
#         else:
#             alternative_nums = []
#             num_increase = dp_temp_reverse[i]  # 以temp_array[i]为最后一位的递增序列的个数
#             for j in range(0, i):
#                 if dp_temp_reverse[j] <= reverse_right_array[i] and dp_temp_reverse[j] == num_increase - 1:
#                     lis = right_alternative_arrays[j]
#                     for item in lis:
#                         li = item + [reverse_right_array[j]]
#                         alternative_nums.append(li)
#             right_alternative_arrays.append(alternative_nums)
#     right_max_alternative_nums = right_alternative_arrays[-1]
#     # print(right_max_alternative_nums)
#     for i in range(len(right_max_alternative_nums)):
#         right_max_alternative_nums[i] = right_max_alternative_nums[i][::-1]
#     # print('右边最长递增子序列候选组：', right_max_alternative_nums)
#     for i in range(len(left_max_alternative_nums)):
#         for j in range(len(right_max_alternative_nums)):
#             a_result_list = left_max_alternative_nums[i] + [temp_array[index_temp]] + right_max_alternative_nums[j]
#             k = 0
#             while k < len(a_result_list):
#                 if k < len(a_result_list) - 1:
#                     print(a_result_list[k], end=' ')
#                 else:
#                     print(a_result_list[k])
#                 k += 1
#             # results.append(a_result_list)
#
#     index += 1
# import copy
# li1 = [[1,2], [3, 4]]
# li2 = copy.deepcopy(li1)
# li1.clear()
#
# print(li2)

# array = [2,43,54,21,8,42, 32,32, 80,23, 80,21,23, 43,53, 13, 32,53]
# array1 = [53, 32, 13, 53, 43, 23, 21,  23, 32, 32, 42, 8, 21, 54, 43, 2]
# input_array = input().split(' ')
# array = [int(item) for item in input_array]
#
#
# def left_to_right(arr):
#     dp = [1] * len(arr)
#     i = 1
#     while i < len(arr):
#         j = 0
#         while j < i:
#             if arr[j] <= arr[i] and 1 + dp[j] > dp[i]:
#                 dp[i] = 1 + dp[i]
#             j += 1
#         i += 1
#     return dp
#
#
# def right_to_left(arr):
#     dp = [1] * len(arr)
#     i = len(arr) - 2
#     while i >= 0:
#         j = len(arr) - 1
#         while j > i:
#             if arr[j] <= arr[i] and 1 + dp[j] > dp[i]:
#                 dp[i] = 1 + dp[i]
#             j -= 1
#         i -= 1
#     return dp
#
#
# def get_max_indexs(arr, dp_1, dp_2):
#     indexs = [0]
#     for i in range(len(arr)):
#         sum = dp_1[i] + dp_2[i]
#         if sum > dp_1[indexs[0]] + dp_2[indexs[0]]:
#             indexs = [i]
#         elif sum == dp_1[indexs[0]] + dp_2[indexs[0]]:
#             indexs.append(i)
#     indexs = list(set(indexs))
#     return indexs
#
#
# def find_remove_indexes(list1, list2):
#     remove_indexes = []
#     remove_alternative_indexes = []
#
#     for i in range(len(list1)):
#         sub_list = [list1[i], list1[i] + 1]
#         remove_alternative_indexes.append(sub_list)
#     for i in range(len(list2)):
#         sub_list = [list2[i], list2[i] + 1]
#         remove_alternative_indexes.append(sub_list)
#
#     # print(remove_alternative_indexes)
#     help_indexes = []
#     for i in range(pow(2, len(remove_alternative_indexes))):
#         help_index_before_process = str(bin(i))[2:]
#         # print(help_index_before_process)
#         if len(help_index_before_process) != len(remove_alternative_indexes):
#             help_index_before_process = '0' * (
#                     len(remove_alternative_indexes) - len(help_index_before_process)) + help_index_before_process
#             help_indexes.append(help_index_before_process)
#         else:
#             help_indexes.append(help_index_before_process)
#     for i in range(len(help_indexes)):
#         help_indexes[i] = list(help_indexes[i])
#         for j in range(len(help_indexes[i])):
#             help_indexes[i][j] = int(help_indexes[i][j])
#     # print(help_indexes)
#     # return remove_alternative_indexes, help_indexs
#     remove_indexes = []
#     if remove_alternative_indexes:
#         for item in help_indexes:
#             sub_indexes = []
#             for i in range(len(item)):
#                 sub_indexes.append(remove_alternative_indexes[i][item[i]])
#             remove_indexes.append(sub_indexes)
#     return remove_indexes
#
#
# dp1 = left_to_right(array)
# dp2 = right_to_left(array)
# indexs = get_max_indexs(array, dp1, dp2)
# # print('indexes:', indexs)
#
# # print(dp1)
# # print(dp2)
# # print(indexs)
# # print()
# results = []
# for index_ in range(len(indexs)):
#     index = indexs[index_]
#     # print('新的情况=======》''index', index, '最大值为', array[index])
#     left_array = array[0: index].copy()
#     right_array = array[index + 1:].copy()
#     # print(left_array)
#     # print(right_array)
#     delta = 0  # index前进的位数
#     for item in left_array:  # 删除大于最终结果最大值的值
#         if item > array[index]:
#             delta += 1
#             left_array.remove(item)
#     for item in right_array:
#         if item > array[index]:
#             right_array.remove(item)
#     temp_array = left_array + [array[index]] + right_array  # 删除大元素后的新数组
#     # print('删除不合格元素后的新数组：', temp_array)
#     # print('删除不符合要求后的array:', temp_array)
#     dp1_temp = left_to_right(temp_array)
#     dp2_temp = right_to_left(temp_array)
#
#     # index_temp = get_max_indexs(temp_array, dp1_temp, dp2_temp)[0]
#     index_temp = index - delta  # 新数组最大数的下标
#     # print('temp_array:', temp_array)
#     # print('index_temp:', index_temp)
#     # print('新数组的dp1', dp1_temp)
#     # print('新数组的dp2', dp2_temp)
#     left_alternative_arrays = []
#     right_alternative_arrays = []
#
#     for i in range(index_temp + 1):
#         if i == 0:
#             left_alternative_arrays.append([[]])
#         else:
#             alternative_nums = []
#             num_increase = dp1_temp[i]  # 以temp_array[i]为最后一位的递增序列的个数
#             for j in range(0, i):
#                 if temp_array[j] <= temp_array[i] and dp1_temp[j] == num_increase - 1:
#                     lis = left_alternative_arrays[j]
#                     for item in lis:
#                         li = item + [temp_array[j]]
#                         alternative_nums.append(li)
#                     # for item in lis:
#                     # li = item + [temp_array[j]]
#                     # li = left_alternative_arrays[j] + [temp_array[j]]
#                     #     alternative_nums.append(li)
#             if len(alternative_nums) == 0:
#                 left_alternative_arrays.append([[]])
#             else:
#                 left_alternative_arrays.append(alternative_nums)
#     left_max_alternative_nums = left_alternative_arrays[-1]
    # print('左边最长递增子序列候选组：', left_max_alternative_nums)

    # i = len(dp2_temp) - 1
    # while i >= index_temp:
    #     if i == len(dp2) - 1:
    #         right_alternative_arrays.append([[]])
    #     else:
    #         alternative_nums = []
    #         num_decrease = dp2_temp[i]
    #         j = len(dp2_temp) - 1
    #         while j > i:
    #             if temp_array[j] <= temp_array[i] and dp2_temp[j] == num_decrease - 1:
    #                 lis = right_alternative_arrays[len(dp2_temp) - j - 1]
    #                 for item in lis:
    #                     li = item + [temp_array[j]]
    #                     alternative_nums.append(li)
    #             j -= 1
    #         right_alternative_arrays.append(alternative_nums)
    # print("将右边的数反转", reverse_right_array)
    #     i -= 1
    # right_max_alternative_nums = right_alternative_arrays[-1]
    # # print('右边最长子序列候选组：', right_max_alternative_nums)
    # reverse_right_array = temp_array[::-1]
    # # print('反转后的新数组：', reverse_right_array)
    # dp_temp_reverse = left_to_right(reverse_right_array)
    # # print('dp反转：',dp_temp_reverse)
    # index_temp_reverse = len(temp_array) - index_temp - 1
    # # print(index_temp_reverse)
    # for i in range(index_temp_reverse + 1):
    #     if i == 0:
    #         right_alternative_arrays.append([[]])
    #     else:
    #         alternative_nums = []
    #         num_increase = dp_temp_reverse[i]  # 以temp_array[i]为最后一位的递增序列的个数
    #         for j in range(0, i):
    #             if reverse_right_array[j] <= reverse_right_array[i] and dp_temp_reverse[j] == num_increase - 1:
    #                 lis = right_alternative_arrays[j]
    #                 for item in lis:
    #                     li = item + [reverse_right_array[j]]
    #                     alternative_nums.append(li)
    #         if len(alternative_nums) == 0:
    #             right_alternative_arrays.append([[]])
    #         else:
    #             right_alternative_arrays.append(alternative_nums)
    # # print(right_alternative_arrays)
    # right_max_alternative_nums = right_alternative_arrays[-1]
    # # print(right_max_alternative_nums)
    # for i in range(len(right_max_alternative_nums)):
    #     right_max_alternative_nums[i] = right_max_alternative_nums[i][::-1]
    # # print('右边各元素最长递增子序列候选组：')
    # # for item in right_alternative_arrays:
    # #     print(item)
    #
    # for i in range(len(left_max_alternative_nums)):
    #     for j in range(len(right_max_alternative_nums)):
    #         a_result_list = left_max_alternative_nums[i] + [temp_array[index_temp]] + right_max_alternative_nums[j]
    #         results.append(a_result_list)
            # k = 0
            # while k < len(a_result_list):
            #     print(a_result_list[k], end=' ')
            #     k += 1
            # # if j < len(right_max_alternative_nums) - 1:
            # if index_ < len(indexs) - 1 or j < len(right_max_alternative_nums) - 1:
            #     print()
    # i = 0
    # j = 0
    # while i < len(left_max_alternative_nums):
    #     while j < len(right_max_alternative_nums):
    #         a_result_list = left_max_alternative_nums[i] + [temp_array[index_temp]] + right_max_alternative_nums[j]
    #         if j == len(right_max_alternative_nums) - 1:
    #             k = 0
    #             while k < len(a_result_list):
    #                 print(a_result_list[k], end=' ')
    #                 k += 1
    #         else:
    #             k = 0
    #             while k < len(a_result_list):
    #                 if k < len(a_result_list) - 1:
    #                     print(a_result_list[k], end=' ')
    #                 else:
    #                     print(a_result_list[k])
    #                 k += 1
    #
    #         j += 1
    #     i += 1
    # index += 1
# print(results, end=' ')
# print(results)
# i = 0
# while i < len(results):
#     if i < len(results) - 1:
#         k = 0
#         while k < len(results[i]):
#             if k < len(results[i]) - 1:
#                 print(results[i][k], end=' ')
#             else:
#                 print(results[i][k])
#             k += 1
#     else:
#         k = 0
#         while k < len(results[i]):
#             print(results[i][k], end=' ')
#             k += 1
#     i += 1

# class Solution:
#     def threeSum(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: List[List[int]]
#         """
#         result = []
#         nums.sort()
#         i = 0
#         while i < len(nums) - 2:
#             j = i + 1
#             if nums[i] + nums[j] > 0:
#                 break
#             while j < len(nums) - 1:
#                 if nums[i] + nums[j] > 0:
#                     break
#                 elif (0 - nums[i] - nums[j]) in nums[j + 1:]:
#                     sub_result = [nums[i], nums[j], 0 - nums[i] - nums[j]]
#                     if sub_result not in result:
#                         result.append(sub_result)
#                 else:
#                     break
#                 j += 1
#             i += 1
#         return result
# s = Solution()
# print(s.threeSum([0, -4, -1, -4, -2, -3, 2]))

# print(abs(55.07718 - 47.929947) // 55.07718)
ARR = [int(x) for x in input().split()]

def longest_inc_seq(arr):
    n = len(arr)
    dp = [1] * n
    lis_arr = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i):
            if arr[i] > arr[j]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    lis_arr[i].append(arr[j])
                if dp[j] + 1 == dp[i] and lis_arr[i][-1] > arr[j]:
                    lis_arr[i][-1] = arr[j]
        lis_arr[i].append(arr[i])
    return dp, lis_arr


def find_longest(arr):
    ldp, lis_arr = longest_inc_seq(arr)
    rdp, lds_arr = longest_inc_seq(arr[::-1])

    rdp = rdp[::-1]
    lds_arr = [e[::-1] for e in lds_arr]
    lds_arr = lds_arr[::-1]

    res = ldp[0] + rdp[0]
    n = 0

    for i in range(len(arr)):
        if res < ldp[i] + rdp[i]:
            res = ldp[i] + rdp[i]
            n = i

    for i in lis_arr[n] + lds_arr[n][1:]:
        print(i, end=' ')

find_longest(ARR)
