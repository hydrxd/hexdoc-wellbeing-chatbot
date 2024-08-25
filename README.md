# HexDoc 
HexDoc is an AI-powered chatbot designed to offer personalized stress relief exercises and mental wellness resources. Utilizing basic Natural Language Processing (NLP), HexDoc interprets user inputs related to mood and stress levels to provide tailored recommendations. The chatbot ensures ease of interaction and engages users in a calming, supportive conversation, helping them manage stress effectively.

# Project Installation 🚀

### Initialization
Create project directory
``` shell
mkdir hexdoc
```
Move to project directory
```
cd hexdoc
```

### Installation
Clone the repository
```
git clone https://https://github.com/lohithgsk/chatbot-rag.git
```
### Usage
Start the flask application on your terminal
```python
flask run
```
The application would be up at ```https://localhost:5000```

# Running the model

# About the model 🤖

This is a feedforward neural network built using TensorFlow's Keras API for classifying user inputs into predefined categories or intents. Here's a brief description:
  - Input Layer: The model takes in a vector of length equal to the number of unique words in the training data (bag-of-words representation).
  - First Hidden Layer: The first layer has 128 neurons with ReLU activation, introducing non-linearity to capture complex patterns in the data. A dropout layer follows, randomly dropping 50% of the neurons during training to prevent overfitting.
  - Second Hidden Layer: This layer has 64 neurons with ReLU activation, followed by another dropout layer, further refining the model's ability to generalize.
  - Output Layer: The output layer has as many neurons as there are classes (intents) in the dataset, with a softmax activation function to output a probability distribution over all possible classes.
  - Compilation: The model is compiled with categorical cross-entropy as the loss function (suitable for multi-class classification) and the Adam optimizer with a learning rate of 0.001, optimizing for accuracy.

This architecture is designed to classify input text into specific categories, such as different types of user intents, by learning from labeled training data.

# Feedback
If you have any feedback you can reach out to us.


