def fibonacci1(n):
    if n == 1:
        return 1
    if n == 2:
        return 1
    return fibonacci1(n - 1) + fibonacci1(n - 2)


# print(fibonacci1(10))

#  自顶向下的备忘录法
def fib(n, mem):

    if mem[n] != -1:
        return mem[n]

    if n == 1 or n == 2:
        mem[n] = 1
    else:
        mem[n] = fib(n - 1, mem) + fib(n - 2, mem)

    return mem[n]


def fibonacci2(n):
    mem = [-1] * (n + 1)
    return fib(n, mem)


# print(fibonacci2(10))

# 自底向上的动态规划法
def fibonacci3(n):

    mem_i_2 = 1
    mem_i_1 = 1
    mem_i = mem_i_1 + mem_i_2
    i = 3
    while i < n:
        mem_i_2 = mem_i_1
        mem_i_1 = mem_i
        mem_i = mem_i_1 + mem_i_2
        i += 1
    return mem_i

# print(fibonacci3(10))