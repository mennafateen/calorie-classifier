import numpy as np

file = open('output.txt').read()
op = open('classes.txt', "w")
lines = file.split('\n')
numbers = []
cl = 0
ch = 0
for line in lines:
    num = float(line)
    if (num < 465):
        #op.write("low\n")
        op.write("0\n")
        cl = cl + 1
    else:
        #op.write("high\n")
        op.write("1\n")
        ch = ch + 1

    numbers.append(num)
print (numbers)
print (cl)
print (ch)
print ("Min: ", min(numbers))
print ("Median: ", np.median(numbers))
print ("Max: ", max(numbers))
