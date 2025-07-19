text = "hello there hello world there world hello there"

import random

def build_markov_chain(text):
    words = text.split()
    markov_chain = {}
    for i in range(len(words)-1):
        curr_word = words[i]
        next_word = words[i+1]
        if curr_word in markov_chain:
            markov_chain[curr_word].append(next_word)
        else:
            markov_chain[curr_word] = [next_word]
    return markov_chain

markov_chain = build_markov_chain(text)
print(markov_chain)

def generate_text(chain, start_word, length=10):
    word = start_word
    output = [word]
    for i in range(length-1):
        next_words = chain.get(word)
        if not next_words:
            break  # stop if no known transition
        word = random.choice(next_words)
        output.append(word)
    return ' '.join(output)

print(generate_text(markov_chain, start_word="hello", length=10))
