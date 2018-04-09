import random
import sys

random.seed(1337)

def do_stuff(f):
	toks = []
	for line in f:
		if line[0] == '#' or not line.strip():
			continue

		cols = line.split("\t")
		if cols[7] == 'case':
			toks.append(cols[1])

	return toks

with open(sys.argv[1]) as f:
	l1 = f.readlines()

with open(sys.argv[2]) as f:
	l2 = f.readlines()

with open(sys.argv[3]) as f:
	combo = f.readlines()

# build samples
toks = do_stuff(l1) + do_stuff(l2)

for line in combo:
	if line[0] == '#' or not line.strip():
		sys.stdout.write(line)
		continue

	cols = line.split("\t")
	if cols[7] == 'case':
		sys.stdout.write("{}\t{}\t{}".format(cols[0], random.choice(toks), "\t".join(cols[2:])))
	else:
		sys.stdout.write(line)

