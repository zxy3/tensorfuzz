import sys
coin = [1, 3, 5]


def min_coins(coin_list, changes):
    if changes <= 0:
        return 0
    sum_list = [[0 for col in range(changes + 1)]for row in range(len(coin_list))]
    for i in range(changes + 1):
        if i % coin_list[0] == 0:
            sum_list[0][i] = i // coin_list[0]
        else:
            sum_list[0][i] = sys.maxsize
    for i in range(len(coin_list)):
        sum_list[i][0] = 0
    i = 1
    while i < len(coin_list):
        j = 1
        while j < (changes + 1):
            if j - coin_list[i] >= 0:
                sum_list[i][j] = min(sum_list[i - 1][j], sum_list[i][j - coin_list[i]] + 1)
            else:
                sum_list[i][j] = sum_list[i - 1][j]
            j += 1
        i += 1
    print(sum_list)
    return sum_list[-1][-1]

print(min_coins([1, 3, 5], 5))