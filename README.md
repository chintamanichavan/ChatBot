```markdown
# SRE Portfolio Chatbot

This is a chatbot designed to provide information on a Site Reliability Engineer (SRE) professional's portfolio. The chatbot is capable of answering queries about the person's skills, projects, experience, and so on.

## Project Structure

- `main.py`: This is where the chatbot's main logic resides. It includes data preprocessing, model training, and the implementation of the chatbot class.
- `model.h5`: This is the trained model that is used for intent classification.
- `tokenizer.pickle`: This is the tokenizer used for preprocessing user input.
- `encoder.pickle`: This is the one-hot encoder used for encoding the intents.

## How to Use

To interact with the chatbot, run `main.py` and follow the prompts. You can ask the chatbot about the SRE professional's skills, projects, and work experience. You can also ask for the professional's contact information.

Example usage:

```python
bot = SREPortfolioBot(model, tokenizer, encoder)
print("Bot:", bot.handle_message('Hello'))
print("Bot:", bot.handle_message('What skills do you have?'))
print("Bot:", bot.handle_message('What projects have you worked on?'))
print("Bot:", bot.handle_message('Tell me about your work experience'))
print("Bot:", bot.handle_message('How can I contact you?'))
print("Bot:", bot.handle_message('Bye'))
```

## Requirements

This project requires Python 3.6+ and the following Python libraries installed:

- numpy
- tensorflow
- sklearn

To run the chatbot, execute the `main.py` script:

```bash
python main.py
```

## Future Work

- Add support for more intents.
- Implement more complex dialogue management.
- Integrate with a messaging platform.
