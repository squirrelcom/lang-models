from random import shuffle

with open('gold') as gold:
    lines = gold.readlines()
    for i, line in enumerate(lines):
        words = line.strip().split(' ')
        jumbles = []
        jumbles.append(' '.join(words))
        for j in xrange(9):
            shuffle(words)
            jumbles.append(' '.join(words))
        with open('test%s' % i, mode='w') as f:
            shuffle(jumbles)
            for word in jumbles:
                f.write(word + '\n')
