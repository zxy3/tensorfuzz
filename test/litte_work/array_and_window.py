array = []
array_input = input().split(' ')
for item in array_input:
    array.append(int(item))

window_size_input = input()
window_size = int(window_size_input)

max_index = 0
i = 0
sum = 0
while i + window_size <= len(array):
    sub_array = array_input[i: i + window_size]
    sub_max = max(sub_array)
    sum += int(sub_max)
    i += 1
print(sum)