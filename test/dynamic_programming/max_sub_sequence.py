# 最长公共子序列递归算法
def LCS1(str1, str2, i, j):
    if i >= len(str1) or j >= len(str2):
        return 0
    if str1[i] == str2[j]:
        return LCS1(str1, str2, i + 1, j + 1) + 1
    else:
        return max(LCS1(str1, str2, i + 1, j), LCS1(str1, str2, i, j + 1))

str1 = list('hello')
str2 = list('welcome')
# print(LCS1(str1, str2, 0, 0))


# 最长公共子序列非递归（动态规划算法）
def LCS2(str1, str2):
    str1, str2 = list(str1), list(str2)
    li = [[0 for col in range(len(str2) + 1)] for row in range(len(str1) + 1)]
    i = 1
    while i < len(li):
        j = 1
        while j < len(li[0]):
            if str1[i - 1] == str2[j - 1]:
                li[i][j] = li[i - 1][j - 1] + 1
            else:
                li[i][j] = max(li[i - 1][j], li[i][j - 1])
            j += 1
        i += 1
    print(li)
    return li[len(str1)][len(str2)]


print(LCS2('hello', 'welcome'))


