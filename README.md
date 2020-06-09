# chitchatbot1
This is a chatbot intended to simulate human chit chat behavior.
The main chatbot code is based on this tutorial posted online:
https://www.youtube.com/watch?v=wypVcNIH6D4
https://techwithtim.net/tutorials/ai-chatbot/part-1/

I give full credit to the original publisher of the code. The code was downloaded and modified for academic purposes. You can copy it and use it for non commercial use. I had to create a unique dataset which consists of tags for conversation, as well as personas to test the hypothesis for some academic papers. The links for the papers are mentioned here:

I. https://arxiv.org/pdf/1801.07243.pdf

II. https://research.fb.com/wp-content/uploads/2018/10/Training-Millions-of-Personalized-Dialogue-Agents.pdf

The main hypothesis is related to personas used for training the model. Usage of personas make the chat more interesting for the end user. I wanted to test this by training a real model with a small dataset consisting of these 4 entities - tag, persona, pattern and response. An example of the dataset is given below:
"tag": "goodbye",
"persona": "Sometimes I act rude to people",
"patterns": ["Hope to never see ya", "Shut up", "Go away"],
"responses": ["Bye and don't come back", "Go away", "Hoping that I don't run into you"]

1. Instructions for running the code:
There are several dependencies that need to be installed along with Python 3.
You should use pip to install the following:
- numpy
- nltk
- tensorflow
- tflearn
The code should work on latest version of Python as of June 2020.

2. After installing all packages, 
For windows - you can open the file train_model.py in VS code and select Python interpreter and then run the code with F5 (Run). Instead, you can run it from command line if python is added to path variable as follow:
python train_model.py

The program should first train the model (if it does not exist in the same location as python code). After training the model, the main chat program starts to converse with the user.

To terminate the program, just type quit or q.

3. Have fun and learn.

