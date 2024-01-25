import tkinter as tk
from tkinter import simpledialog
import numpy as np
import random
import json
import re

# Load predefined patterns and responses from intents.json
with open('intents.json') as file:
    data = json.load(file)

# Simple tokenizer
# Splits sentences into words
def tokenize(sentence):
    return re.findall(r'\w+', sentence.lower())

# Stemming function
# Convert to lowercase
def stem(word):
    return word.lower()

# Preprocess the data
# store the stemmed words and their corresponding labels.
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

# Function to get the response from the bot
def get_response(inp):
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
        return "I don't understand, please try again."

    responses = [response for intent in data["intents"] if intent["tag"] == best_match for response in intent["responses"]]
    return random.choice(responses)

# GUI part
class ChatBotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ChatBot GUI")
        self.geometry("400x500")

        self.text_widget = tk.Text(self, state='disabled')
        self.text_widget.pack(padx=15, pady=15)

        self.entry = tk.Entry(self)
        self.entry.pack(padx=15, pady=15)

        self.send_button = tk.Button(self, text="Send", command=self.send_message)
        self.send_button.pack(padx=15, pady=5)

    def send_message(self):
        user_input = self.entry.get()
        self.entry.delete(0, tk.END)

        self.update_chat_window("You: " + user_input)
        response = get_response(user_input)
        self.update_chat_window("Bot: " + response)

    def update_chat_window(self, message):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, message + "\n")
        self.text_widget.configure(state='disabled')
        self.text_widget.see(tk.END)

if __name__ == "__main__":
    app = ChatBotApp()
    app.mainloop()
