import numpy as np
import random
import json
import re

# Load predefined patterns and responses from intents.json
with open('intents.json') as file:
    data = json.load(file)

# Simple tokenizer
def tokenize(sentence):
    return re.findall(r'\w+', sentence.lower())

# Stemming function (basic version)
def stem(word):
    return word.lower()

# Preprocess the data
words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = tokenize(pattern)
        words.extend(wrds)
        docs_x.append([stem(w) for w in wrds])
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = sorted(list(set(words)))
labels = sorted(labels)

# Convert text to numerical data (bag of words)
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = [stem(w) for w in tokenize(s)]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# Chatbot response logic
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = bag_of_words(inp, words)

        # Find the best match
        best_match = None
        highest_prob = 0
        for idx, pattern_bag in enumerate(docs_x):
            similarity = np.dot(results, np.array(bag_of_words(' '.join(pattern_bag), words)))
            if similarity > highest_prob:
                highest_prob = similarity
                best_match = docs_y[idx]

        if best_match is None:
            print("I don't understand, please try again.")
            continue

        responses = [response for intent in data["intents"] if intent["tag"] == best_match for response in intent["responses"]]
        print(random.choice(responses))

if __name__ == "__main__":
    chat()
