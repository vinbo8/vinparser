import sys
import random

random.seed(1337)

seedlist = []
with open(sys.argv[1], "r") as f:
    for line in f:
        seedlist.append(tuple(line.rstrip("\n").split("\t")))

a_to_b = {k: v for (k, v) in seedlist}
b_to_a = {v: k for (k, v) in seedlist}

blokk, edited_in_sentence, offset = [], [], 0
for line in sys.stdin:
    if not line.rstrip():
        for edited_id in edited_in_sentence:
            head = int(blokk[edited_id][6]) - 1

            # swap
            temp = blokk[head]
            blokk[head] = blokk[edited_id]
            blokk[edited_id] = temp

            # fix IDs
            new_id = blokk[head][0]
            old_id = blokk[edited_id][0]
            blokk[head][0] = blokk[edited_id][0]
            blokk[edited_id][0] = new_id

            # fix other rels
            for blokk_line in blokk:
                if blokk_line[6] == new_id:
                    blokk_line[6] = old_id
                elif blokk_line[6] == old_id:
                    blokk_line[6] = new_id

        for blokk_line in blokk:
            try:
                sys.stdout.write("\t".join(blokk_line))
            except:
                print(blokk_line)
                sys.exit()
        blokk, edited_in_sentence, offset = [], [], 0
        sys.stdout.write(line)

    elif line[0] == '#':
        sys.stdout.write(line)

    else:
        cols = line.split("\t")
        if "." in cols[0] or "-" in cols[0]:
            offset += 1
        if cols[7] == 'case':
            flip = random.randint(0, 3) == 0
            if flip:
                # subtract one to get index in blokk
                form, head = cols[1], (int(cols[6]) - 1)
                if form in a_to_b:
                    cols[1] = a_to_b[form]
                    edited_in_sentence.append(int(cols[0]) - 1 + offset)
                elif form in b_to_a:
                    cols[1] = b_to_a[form]
                    edited_in_sentence.append(int(cols[0]) - 1 + offset)

        blokk.append(cols)