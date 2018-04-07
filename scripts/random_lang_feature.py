import sys
import time
import random

random.seed(time.time())

for line in sys.stdin:
	if line[0] == '#':
		sys.stdout.write(line)
		continue
	
	if not line.strip():
		sys.stdout.write(line)
		continue
	
	cols = line.split("\t")
	sys.stdout.write("{}\t{}\t{}\t{}\t{}\t".format(*cols[0:5]))
	sys.stdout.write("{}\t{}\t{}\t{}\tLanguage={}\n".format(*cols[5:9], random.choice(sys.argv[1:])))
