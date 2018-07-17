import sys
import random

random.seed(1337)

for line in sys.stdin:
    if line[0] == '#' or not line.strip():
        sys.stdout.write(line)

    else:
        cols = line.split("\t")
        cols[9] = random.choice(["_\n", "rus\n"])
        sys.stdout.write("\t".join(cols))
