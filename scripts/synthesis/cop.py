from collections import Counter
import random
import sys

random.seed(1337)

for line in sys.stdin:
	if line[0] == '#' or not line.strip():
		sys.stdout.write(line)
		continue

	cols = line.split("\t")
	lemma, deprel = cols[1], cols[7]
	if lemma in ["is", "'s"] and deprel == 'cop':
		if random.randint(0, 1):
			cols[1] = 'है'

	if lemma == 'है' and deprel == 'cop':
		if random.randint(0, 1):
			cols[1] = 'is'

	sys.stdout.write("\t".join(cols))
