try:
    while True:
        s = input()
        items = s.split(' ')
        a = int(items[0])
        b = int(items[1])
        print(a + b)
except EOFError:
    pass

# =======================================================.
input_lines = int(input())
i = 0
while i < input_lines:
    line = input().split(' ')
    print(int(line[0]) + int(line[1]))
    i += 1

#=======================================================

try:
  while True:
    s = input()
    items = s.split(' ')
    if int(items[0]) == 0 and int(items[1]) == 0:
      break;
    print(int(items[0]) + int(items[1]))

except EOFError:
  pass

#=======================================================

while True:
    items = input().split(' ')
    if int(items[0]) == 0:
        break
    else:
        result = 0
        number = int(items[0])
        index = 1
        while index < 1 + number:
            result += int(items[index])
            index += 1
        print(result)

#=======================================================

line_number = int(input())
line = 0
while line < line_number:
  items = input().split(' ')
  index = 1
  result = 0
  while index < 1 + int(items[0]):
    result += int(items[index])
    index += 1
  print(result)
  line += 1


#=======================================================

lines = 2
idx = 0
while idx < lines:
    items = input().split(' ')
    number = int(items[0])
    result = 0
    index = 1
    while index <= number:
        result += int(items[index])
        index += 1
    print(result)
    idx += 1

# =======================================================

lines = 2
idx = 0
while idx < lines:
        items = input().split(' ')
        result = 0
        for item in items:
            result += int(item)
        print(result)
        print()
        idx += 1

# =======================================================

line_nubmer = int(input())
index = 0
while index < line_nubmer:
    items = input().split(' ')
    number = int(items[0])
    idx = 1
    result = 0
    while idx <= number:
        result += int(items[idx])
        idx += 1
    print(result)
    print()
    index += 1

