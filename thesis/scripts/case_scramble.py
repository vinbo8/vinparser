import sys
import random

tokens = [[], []]
for i in [0, 1]: 
    with open(sys.argv[i + 1], "r") as f:
        for line in f:
            if line[0] == '#' or not line.strip():
                continue

            cols = line.split("\t")
            form, deprel = cols[1], cols[7]
            if deprel == 'case':
                # TODO: compare this with sets
                tokens[i].append(form)

for line in sys.stdin:
    if line[0] == '#' or not line.strip():
        sys.stdout.write(line)

    else:
        cols = line.split("\t")
        if cols[7] == 'case':
            flip = random.randint(0, 3) == 0
            if flip:
                form = cols[1]
                if form in tokens[0]:
                    cols[1] = random.choice(tokens[1])
                elif form in tokens[1]:
                    cols[1] = random.choice(tokens[0])
                else:
                    continue

        sys.stdout.write("\t".join(cols))