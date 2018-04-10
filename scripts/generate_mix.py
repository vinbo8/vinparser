from collections import Counter
import random
import sys

random.seed(1337)

c = Counter()

def do_stuff(f):
	global c
	toks = []
	for line in f:
		if line[0] == '#' or not line.strip():
			continue

		cols = line.split("\t")
		if cols[7] == 'case':
			c[cols[1]] += 1

with open(sys.argv[1]) as f:
	l1 = f.readlines()

with open(sys.argv[2]) as f:
	l2 = f.readlines()

with open(sys.argv[3]) as f:
	combo = f.readlines()

# build samples
do_stuff(l1)
do_stuff(l2)


toks = [item for sublist in c for item in sublist]

for line in combo:
	if line[0] == '#' or not line.strip():
		sys.stdout.write(line)
		continue

'''
	cols = line.split("\t")
	if cols[7] == 'case':
		shuf = random.randint(1, 4)
		if shuf == 1:
			sys.stdout.write("{}\t{}\t{}".format(cols[0], random.choice(toks), "\t".join(cols[2:])))
		else:
			sys.stdout.write(line)
	else:
		sys.stdout.write(line)
'''
