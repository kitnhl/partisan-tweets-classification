import sys

fil = sys.argv[1]
csvfilename = open(fil, 'r').readlines()
number = 1
for j in range(len(csvfilename)):
    if j % 160000 == 0:
	   open(str(fil) + str(number) + '.csv', 'w+').writelines(csvfilename[j:j+160000])
	   number += 1