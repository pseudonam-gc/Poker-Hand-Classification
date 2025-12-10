import random

augment_rare_classes = True
avoid_permutation_overlap = True
if augment_rare_classes:
    output_file = "poker/all-poker-hands-augmented"
else:
    output_file = "poker/all-poker-hands"

if avoid_permutation_overlap:
    output_file += "-perm-free"

output_test_file = output_file + "-test.data"
output_train_file = output_file + "-train.data"
train_split = 0.8

# generates training data 
numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13]
suits = [1,2,3,4]
cards = [(n,s) for n in numbers for s in suits]

import itertools 

def hand_class(hand):
    # hand is a list of 5 (number,suit) tuples
    suits = [s for n,s in hand]
    numbers = sorted([n for n,s in hand])
    is_flush = len(set(suits))==1
    is_straight = (len(set(numbers)) == 5 and (numbers[-1]-numbers[0] == 4 or numbers == [1,10,11,12,13]))
    if is_straight and is_flush and numbers == [1,10,11,12,13]:
        return 9
    if is_straight and is_flush:
        return 8
    if len(set(numbers)) == 2:
        # either four of a kind or full house 
        if numbers[0] == numbers[1] and numbers[3] == numbers[4]:
            return 6  # full house
        else:
            return 7
    if is_flush:
        return 5 
    if is_straight:
        return 4
    if len(set(numbers)) == 3:
        # either three of a kind or two pair
        if numbers[0] == numbers[1] == numbers[2] or numbers[1] == numbers[2] == numbers[3] or numbers[2] == numbers[3] == numbers[4]:
            return 3
        return 2
    if len(set(numbers)) == 4:
        return 1
    return 0

"""
0: Nothing in hand; not a recognized poker hand

1: One pair; one pair of equal ranks within five cards

2: Two pairs; two pairs of equal ranks within five cards

3: Three of a kind; three equal ranks within five cards

4: Straight; five cards, sequentially ranked with no gaps

5: Flush; five cards with the same suit

6: Full house; pair + different rank three of a kind

7: Four of a kind; four equal ranks within five cards

8: Straight flush; straight + flush

9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush
"""

def test_classifier(src = "poker/poker-hand-training-true.data"):
    with open(src,"r") as f:    
        for line in f.readlines():
            parts = line.strip().split(",")
            hand = []
            for i in range(0,10,2):
                s = int(parts[i])
                n = int(parts[i+1])
                hand.append((n,s))
            cls = int(parts[-1])
            pred = hand_class(hand)
            if cls != pred:
                print (f"Error: got {pred} expected {cls} for hand {hand}")
                return 0
            
    print ("All classification tests passed!")

test_classifier()

train_outputs = []
test_outputs = []
freqs = [0 for _ in range(10)]
for hand in itertools.combinations(cards,5):
    c = hand_class(hand)
    freqs[c] += 1
total = sum(freqs)

print ("Hand class frequencies") 
for i in range(10):
    print (f"Class {i} count: {freqs[i]}")

# randomly order hand
final_freqs = [0 for _ in range(10)]
counter = 0

rng_shuffle = random.Random() # handles shuffling
rng_split = random.Random() # handles train/test assignments

for hand in itertools.combinations(cards,5):

    c = hand_class(hand)
    repeat = 5

    if augment_rare_classes:
        repeat += int(total/(10*freqs[c]))

    for _ in range(repeat+1):
        final_freqs[c] += 1
        counter += 1
        if counter % 100000 == 0:
            print (f"Generated {counter} hands...")

        ordered_hand = list(hand)
        rng_shuffle.shuffle(ordered_hand)

        s = ",".join([f"{s},{n}" for n,s in ordered_hand])    

        # hash of ordered_hand to re-seed random
        # this is more efficient than making a hashmap of hands which have been seen of size 2e7
        if avoid_permutation_overlap:
            hand_hash = hash(tuple(sorted(ordered_hand)))
        else:
            hand_hash = hash(tuple(ordered_hand))
        rng_split.seed(hand_hash)

        if rng_split.random() < train_split:
            train_outputs.append(f"{s},{c}\n")
        else:
            test_outputs.append(f"{s},{c}\n")

for i in range(10):
    print (f"Class {i} count: {final_freqs[i]}")

print (f"Total hands generated: {counter}")
print ("Generated all outputs.")

with open(output_train_file,"w") as f:
    f.writelines(train_outputs)
with open(output_test_file,"w") as f:
    f.writelines(test_outputs)
print ("Finished writing to output " + output_train_file + " and " + output_test_file)