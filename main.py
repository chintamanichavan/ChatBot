# Import necessary libraries
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
import random

# Define the training data
data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "Hey"],
            "responses": ["Hello! How can I assist you with the SRE portfolio today?"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you", "Goodbye"],
            "responses": ["Goodbye! If you have any more questions, feel free to ask."],
        },
        {
            "tag": "skills",
            "patterns": ["What skills do you have?", "Tell me about your skills", "What are your main skills?", "What technical skills do you possess?"],
            "responses": ["I have experience with system design, developing and maintaining CI/CD pipelines, incident response, and I'm proficient in Python, Go, and Shell scripting. I'm also experienced with tools like Docker, Kubernetes, and Terraform."],
        },
        {
            "tag": "projects",
            "patterns": ["What projects have you worked on?", "Tell me about your projects", "What major projects have you completed?"],
            "responses": ["I have worked on several projects including developing an auto-remediation system, implementing a log-monitoring system, and optimizing CI/CD pipelines for several major services."],
        },
        {
            "tag": "experience",
            "patterns": ["What experience do you have?", "Tell me about your work experience", "What is your experience in SRE?"],
            "responses": ["I have 5 years of experience as an SRE, working in fast-paced environments and ensuring the reliability of scalable, distributed systems."],
        },
        {
            "tag": "contact",
            "patterns": ["How can I contact you?", "What's your contact info?", "How to get in touch with you?"],
            "responses": ["You can reach me at my email: sre_professional@example.com"],
        },
    ]
}

# Preprocessing
patterns = [item for intent in data['intents'] for item in intent['patterns']]
tags = [intent['tag'] for intent in data['intents'] for _ in intent['patterns']]
tokenizer = Tokenizer(lower=True, oov_token='oov')
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, padding='post')

# One-hot encode tags
encoder = OneHotEncoder(sparse=False)
tags = np.array(tags).reshape(-1, 1)
encoded_tags = encoder.fit_transform(tags)

# Define and train the model
model = Sequential()
model.add(Embedding(10000, 16, input_length=padded_sequences.shape[1]))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(encoded_tags.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.fit(padded_sequences, encoded_tags, epochs=200)

class SREPortfolioBot:
    def __init__(self, model, tokenizer, encoder):
        self.state = 'INITIAL'
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder

    def predict_intent(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        sequence = pad_sequences(sequence, padding='post')
        prediction = self.model.predict(sequence)
        tag = self.encoder.inverse_transform(prediction)[0][0]
        return tag

    def handle_intent(self, intent):
        if self.state == 'INITIAL' and intent == 'greeting':
            self.state = 'GREETED'
            return "Hello! How can I assist you with the SRE portfolio today?"
        elif self.state in ['GREETED', 'ASKED_SKILLS', 'ASKED_PROJECTS', 'ASKED_EXPERIENCE'] and intent == 'skills':
            self.state = 'ASKED_SKILLS'
            return "I have experience with system design, developing and maintaining CI/CD pipelines, incident response, and I'm proficient in Python, Go, and Shell scripting. I'm also experienced with tools like Docker, Kubernetes, and Terraform."
        elif self.state in ['GREETED', 'ASKED_SKILLS', 'ASKED_PROJECTS', 'ASKED_EXPERIENCE'] and intent == 'projects':
            self.state = 'ASKED_PROJECTS'
            return "I have worked on several projects including developing an auto-remediation system, implementing a log-monitoring system, and optimizing CI/CD pipelines for several major services."
        elif self.state in ['GREETED', 'ASKED_SKILLS', 'ASKED_PROJECTS', 'ASKED_EXPERIENCE'] and intent == 'experience':
            self.state = 'ASKED_EXPERIENCE'
            return "I have 5 years of experience as an SRE, working in fast-paced environments and ensuring the reliability of scalable, distributed systems."
        elif self.state in ['GREETED', 'ASKED_SKILLS', 'ASKED_PROJECTS', 'ASKED_EXPERIENCE'] and intent == 'contact':
            self.state = 'ASKED_CONTACT'
            return "You can reach me at my email: sre_professional@example.com"
        elif self.state != 'INITIAL' and intent == 'goodbye':
            self.state = 'INITIAL'
            return "Goodbye! If you have any more questions, feel free to ask."
        else:
            return "Sorry, I didn't understand that. Can you please rephrase?"

    def handle_message(self, message):
        intent = self.predict_intent(message)
        response = self.handle_intent(intent)
        return response
