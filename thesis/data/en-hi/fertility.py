import sys

total, cs = 0, 0

current_blokk = []
for line in sys.stdin:
	if line[0] == '#':
		continue

	elif not line.rstrip():
		for word in current_blokk:
			head = int(word[6]) - 1
			lang = word[9].rstrip("\n")
			if head != 0:
				head_lang = current_blokk[head][9].rstrip("\n")
				total += 1
				if head_lang != lang:
					cs += 1

		current_blokk = []
		continue

	else:
		current_blokk.append(line.split("\t"))

print(total, cs)

	
		
