
def edit_distance(str1, str2):
    dp = [[0 for col in range(len(str2) + 1)] for row in range(len(str1) + 1)]

    for i in range(len(str1) + 1):
        for j in range(len(str2) + 1):
            if i == 0 and j == 0:
                dp[i][j] = 0
            elif i == 0 and j > 0:
                dp[i][j] = j
            elif j == 0 and i > 0:
                dp[i][j] = i
            else:
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1] + 1, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    return dp[len(str1)][len(str2)]

print(edit_distance('cfe', 'coffe'))