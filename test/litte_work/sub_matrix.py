matrix = []

# while True:
#     row_input = input()
#     if row_input == '':
#         break
#     else:
#         row_input_split = row_input.split(' ')
#         row = []
#         for item in row_input_split:
#             row.append(int(item))
#         matrix.append(row)
try:
    while True:
        row_input = input()
        # if row_input == '':
        #     break
        # else:
        row_input_split = row_input.split(' ')
        row = []
        for item in row_input_split:
            row.append(int(item))
        matrix.append(row)
except EOFError:
    pass
print(matrix)
i = 0
j = 0
max = 0
while i < len(matrix):
    j = 0
    while j < len(matrix[0]):
        if matrix[i][j] == 0:
            j += 1
        else:  #  如果matrix[i][j]=0，设当前子矩阵为1*1
            width = 1
            height = 1

            j1 = j + 1
            while j1 < len(matrix[0]):  # 先向右扩展
                if matrix[i][j1] == 1:
                    width += 1
                else:
                    break
                j1 += 1

            delta_width = 1
            while delta_width <= width:
                height = 1
                i1 = i + 1
                while i1 < len(matrix):  # 再向下扩展
                    sub_matrix = matrix[i1][j: j + delta_width]
                    if all(sub_matrix) == 1:
                        height += 1
                        i1 += 1
                    else:
                        break
                sub_num = delta_width * height  # 计算当前子矩阵中1的个数
                # print(i, " ", j, " ", "width:", delta_width, " height:", height, " ", sub_num)
                if sub_num > max:
                    max = sub_num

                delta_width += 1
            j += 1
    i += 1
print(max)
