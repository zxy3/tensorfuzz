def max_sub_common_string(str1, str2):
    li = [[0 for col in range(len(str2))] for row in range(len(str1))]
    str1, str2 = list(str1), list(str2)
    max = li[0][0]
    for i in range(len(li)):
        for j in range(len(li[0])):
            # print(str1[i], str2[j])
            if str1[i] == str2[j]:
                li[i][j] = li[i - 1][j - 1] + 1
                if li[i][j] > max:
                    max = li[i][j]
    return max


print(max_sub_common_string('hello', 'wellocome'))
