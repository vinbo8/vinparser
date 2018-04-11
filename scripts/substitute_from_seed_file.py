import sys

converts = {}
with open(sys.argv[1], 'r') as seed:
	seed = seed.readlines()
	for line in seed:
		if line[0] == '#':
			continue
		
		converts[line.split("\t")[0]] = line.split("\t")[1].rstrip("\n")


reverse = {converts[k]: k for k in converts}
converts = {**converts, **reverse}

for line in sys.stdin:
	if line[0] == '#' or not line.strip():
		sys.stdout.write(line)
		continue

	cols = line.split("\t")
	if cols[1] in converts:
		cols[1] = converts[cols[1]]

	sys.stdout.write("\t".join(cols))
	
