import sys

for line in sys.stdin:
	if line[0] == '#':
		sys.stdout.write(line)
		continue

	if not line.strip():
		sys.stdout.write(line)
		continue

	cols = line.split("\t")
	sys.stdout.write("{}\t{}\t{}\t{}\t{}\t".format(cols[0], cols[2], cols[1], cols[3], cols[4]))
	sys.stdout.write("{}\t{}\t{}\t{}\tLanguage={}\n".format(cols[5], cols[6], cols[7], '_', cols[8]))
	

