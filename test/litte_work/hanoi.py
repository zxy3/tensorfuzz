num_input = input()
number = int(num_input)


def process(num, left, mid, right, from_, to_):
    if num == 1:
        if from_ == mid or to_ == mid:
            # print("Move 1 from ", from_, " to ", to_)
            return 1
        else:
            # print("Move 1 from ", from_, " to ", mid)
            # print("Move 1 from ", mid, " to ", to_)
            return 2

    if from_ == mid or to_ == mid:
        another = right if (from_ == left or to_ == left) else left
        part1 = process(num - 1, left, mid, right, from_, another)
        part2 = 1
        # print("Move ", num, " from ", from_, " to ", to_)
        part3 = process(num - 1, left, mid, right, another, to_)
        return part1 + part2 + part3
    else:  # 第一步
        part1 = process(num - 1, left, mid, right, from_, to_)
        part2 = 1
        # print("Move ", num, " from ", from_, " to ", mid)
        part3 = process(num - 1, left, mid, right, to_, from_)
        part4 = 1
        # print("Move ", num, " from ", mid, " to ", to_)
        part5 = process(num - 1, left, mid, right, from_, to_)
        return part1 + part2 + part3 + part4 + part5


def hanoi_problem(num, left, mid, right):
    if num < 1:
        return 0
    return process(num, left, mid, right, left, right)


steps = hanoi_problem(number, "left", "mid", "right")
print(int(steps))

