def create_biagrams(word):
    biagrams = []
    for i in range(len(word) - 1):
        biagrams.append(word[i:i+2])
    return biagrams

def create_biagrams_with_gap(word):
    biagrams = []
    for i in range(len(word)):
        biagrams.append(word[:i] + word[i+1:])
    return biagrams

def get_similarity_ratio(word1, word2):
    word1, word2 = word1.lower(), word2.lower()

    common = []

    biagram1 = create_biagrams(word1)
    biagram2 = create_biagrams(word2)
    biagram2_with_gap = create_biagrams_with_gap(word2)

    for bg1 in biagram1:
        if bg1 in biagram2:
            common.append(bg1)
        elif bg1 in biagram2_with_gap:
            common.append(bg1)


    return len(common) / max(len(biagram1), len(biagram2))

def auto_correct(word, database, sim_threshold=0.49):
    if len(word) <= 1:  # Account for empty or single letter words
        return word

    max_sim = 0.0
    most_sim_word = word

    for d_word in database:
        cur_sim = get_similarity_ratio(word, d_word)
        if cur_sim > max_sim:
            max_sim = cur_sim
            most_sim_word = d_word

    return most_sim_word if max_sim > sim_threshold else word

def combine_to_list(words):
    return ' '.join(words)