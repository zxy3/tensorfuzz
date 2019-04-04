price = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]


# 递归方法
def cut1(price_list, n):
    if n == 0:
        return 0
    cost = 0
    i = 1
    while i <= n:
        cost = max(cost, price_list[i] + cut1(price_list, n - i))
        i += 1
    return cost


# print(cut1(price, 4))

# 备忘录版本
def cut_memo(price_list, n):
    memo = [-1] * len(price_list)
    return cut2(price_list, n, memo)


def cut2(price_list, n, memo):
    cost = -1
    if memo[n] >= 0:
        return memo[n]
    if n == 0:
        cost = 0
    else:
        i = 1
        while i <= n:
            cost = max(cost, price_list[i] + cut2(price_list, n - i, memo))
            i += 1
    return cost

# print(cut_memo(price, 4))


# 自底向上的动态规划
def button_up_cut(price_list, n):

    cost_list = [0] * (n + 1)
    i = 1
    while i <= n:
        cost = -1
        j = 1
        while j <= i:
            cost = max(cost, price_list[j] + cost_list[i - j])
            j += 1
        cost_list[i] = cost
        i += 1
    return cost_list[n]


print(button_up_cut(price, 4))