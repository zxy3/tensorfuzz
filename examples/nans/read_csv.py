import csv
output_data = []
with open('output_boundary.csv') as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    birth_header = next(csv_reader)  # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        output_data.append(row)
output_data = [[float(item) for item in row] for row in output_data]
print(output_data)