# Solution generated by ChatGPT <https://chat.openai.com/chat>

# The code first reads the sequences from the file and stores them in a list
# of lists, where each inner list represents a sequence of words.

# Then, it iterates over each sequence, finds all indices of word1 and word2
# in the sequence, and calculates the absolute difference between each pair
# of indices as the distance between the two words in that sequence.
# If the distance is smaller than the current minimum distance, it updates
# the minimum distance and the sequence number.

# Finally, if a minimum distance was found, it prints the sequence number
# and the distance. Otherwise, it prints a message indicating that the two
# words never appear in the same sequence.


def get_min_distance(file, word1, word2):
    sequences = []
    with open(file, 'r') as f:
        for line in f:
            sequences.append(line.strip().split())

    min_distance = float('inf')
    min_seq_num = None
    for i, seq in enumerate(sequences):
        word1_indices = [j for j, x in enumerate(seq) if x == word1]
        word2_indices = [j for j, x in enumerate(seq) if x == word2]
        for index1 in word1_indices:
            for index2 in word2_indices:
                distance = abs(index1 - index2)
                if distance < min_distance:
                    min_distance = distance
                    min_seq_num = i
    return min_seq_num, min_distance


file = 'seq.txt'
word1 = input('Enter the first word: ')
word2 = input('Enter the second word: ')

result = get_min_distance(file, word1, word2)
if result[0] is not None:
    seq_num, distance = result
    print(f'Min distance: sequence {seq_num + 1} (distance={distance})')
else:
    print('The two words never appear in the same sequence')
