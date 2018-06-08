import sys
import random

random.seed(42)

for line in sys.stdin:
    if line[0] == '#' or not line.strip():
        sys.stdout.write(line)

    else:
        cols = line.split("\t")
        cols[9] = random.choice(["Language=hi\n", "Language=en\n"])
        sys.stdout.write("\t".join(cols))