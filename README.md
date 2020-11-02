# Twitter_sentiment
**Version 1.0.0** <br /> 
This is  twiiter sentiment analysis project.Here i have taken tweets related to corna virus and analyzed them first using text blob. <br /> 
Then i have created my own model as according to textblob a tweet with:<br /> 
"I have been tested postive for corona"<br /> 
is a neutral tweet.<br /> 
Well it is not moreover it is a negative tweet.<br /> 
## Installation
I have used anconda,collab and jupyter notebook so if you are using anaconad ypu need to install the following
First I have created a enivorment in anaconda as i as using Python 3.8 and tensor flow supports in 3.7 and 
below in anaconda
So open the anaconda and runn the following command
```bash
   conda create -n myenv python=3.5 
```
This creates a python environement named myenv  with version of python 3.5 <br /> 
Packages to be installed:-
```bash
   conda install -c conda-forge tweepy
   conda install -c anaconda pandas
   conda install -c conda-forge wordcloud
   conda install -c anaconda numpy
   conda install -c anaconda regex
   conda install -c conda-forge textblob
   conda install -c conda-forge matplotlib
```
These packages are used to analysis using textblob.<br />
To do the sentiment analysis using my model I suggest you to use Collab<br />
Packages for model:
```bash
   conda install -c conda-forge spacy
   conda install -c anaconda tensorflow
   conda install -c conda-forge spacy-model-en_core_web_sm
```
I am assuming you have also installled packages from list-1<br />
You also need twiiters api for downloading the tweets for that you need to create<br />
a twiiter's developer account <br />
You can follow the video in link below to get your api: <br />
 [Twitter API video!](https://pythonprogramming.net/twitter-api-streaming-tweets-python-tutorial/)
## Working
**Basic using textblob**<br />
I have created three different notebooks in this project first  i have done the analysis using textblob which is pretty simple and easy you just need basic 
knowledge on how to use panda,textblob and matplot.lib <br />
**About the model that i created**<br />
Now for the model I have first downloaded tweets using tweepy and save them in csv file<br /> 
And i have manually given sentiment to these 200 tweets by reading them<br />
0 is a negative tweet and 1 is a positive tweet<br />
  **Data Cleaning**<br />
  Later on after saving and editing them I have applied data cleaning <br />
  I have removed stopwords and punctiouns using spacy and used my custom<br />
  data cleaning function to remove #,@,emojis,Urls etc<br/>
  **Tokenizing**<br />
Tokenization is a way of separating a piece of text into smaller units called tokens. Here, tokens can be either words, characters, or subwords. Hence, tokenization can be broadly classified into 3 types – word, character, and subword (n-gram characters) tokenization.<br />
<br />
For example, consider the sentence: “Never give up”.<br />
<br />
The most common way of forming tokens is based on space. Assuming space as a delimiter, the tokenization of the sentence results in 3 tokens – Never-give-up. As each token is a word, it becomes an example of Word tokenization.<br />
You can read more  [here](https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/)<br />
<br />
**Word2vec**<br />
Word2vec is a technique for natural language processing. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. As the name implies, word2vec represents each distinct word with a particular list of numbers called a vector. The vectors are chosen carefully such that a simple mathematical function (the cosine similarity between the vectors) indicates the level of semantic similarity between the words represented by those vectors.
![alt text](https://i.stack.imgur.com/N81kM.png)<br />
 **About the model**<br />
 I have used 4 layers in my nerual network model.First I have used <br />
 a basic nembeding layer that i made using word2vec after that I have used <br />
 GlobalMAxpool1D and 2 dense layer with differnet activation one with relu and <br />
 other with sigmod
```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 100),
   
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
 **Accuracy**
 The model gave an accuracy of 92% over training and 88% on testing 
 ```
 history = model.fit(ctweets_test, sentiments_test, epochs=20, validation_data=(ctweets_test, sentiments_test), verbose=1)

Epoch 1/20
2/2 [==============================] - 0s 87ms/step - loss: 0.6913 - accuracy: 0.6000 - val_loss: 0.6848 - val_accuracy: 0.6000
Epoch 2/20
2/2 [==============================] - 0s 39ms/step - loss: 0.6845 - accuracy: 0.6000 - val_loss: 0.6790 - val_accuracy: 0.6000
Epoch 3/20
2/2 [==============================] - 0s 36ms/step - loss: 0.6783 - accuracy: 0.6000 - val_loss: 0.6729 - val_accuracy: 0.6000
Epoch 4/20
2/2 [==============================] - 0s 36ms/step - loss: 0.6722 - accuracy: 0.6000 - val_loss: 0.6668 - val_accuracy: 0.6000
Epoch 5/20
2/2 [==============================] - 0s 35ms/step - loss: 0.6664 - accuracy: 0.6000 - val_loss: 0.6606 - val_accuracy: 0.6000
Epoch 6/20
2/2 [==============================] - 0s 36ms/step - loss: 0.6600 - accuracy: 0.6000 - val_loss: 0.6542 - val_accuracy: 0.6000
Epoch 7/20
2/2 [==============================] - 0s 38ms/step - loss: 0.6534 - accuracy: 0.6000 - val_loss: 0.6478 - val_accuracy: 0.6000
Epoch 8/20
2/2 [==============================] - 0s 38ms/step - loss: 0.6471 - accuracy: 0.6000 - val_loss: 0.6413 - val_accuracy: 0.6000
Epoch 9/20
2/2 [==============================] - 0s 37ms/step - loss: 0.6408 - accuracy: 0.6000 - val_loss: 0.6344 - val_accuracy: 0.6250
Epoch 10/20
2/2 [==============================] - 0s 37ms/step - loss: 0.6336 - accuracy: 0.6250 - val_loss: 0.6269 - val_accuracy: 0.6250
Epoch 11/20
2/2 [==============================] - 0s 38ms/step - loss: 0.6261 - accuracy: 0.6250 - val_loss: 0.6191 - val_accuracy: 0.6500
Epoch 12/20
2/2 [==============================] - 0s 37ms/step - loss: 0.6181 - accuracy: 0.6500 - val_loss: 0.6107 - val_accuracy: 0.7000
Epoch 13/20
2/2 [==============================] - 0s 41ms/step - loss: 0.6101 - accuracy: 0.7000 - val_loss: 0.6019 - val_accuracy: 0.7000
Epoch 14/20
2/2 [==============================] - 0s 37ms/step - loss: 0.6009 - accuracy: 0.7000 - val_loss: 0.5925 - val_accuracy: 0.7250
Epoch 15/20
2/2 [==============================] - 0s 39ms/step - loss: 0.5914 - accuracy: 0.7500 - val_loss: 0.5826 - val_accuracy: 0.8000
Epoch 16/20
2/2 [==============================] - 0s 37ms/step - loss: 0.5817 - accuracy: 0.8000 - val_loss: 0.5724 - val_accuracy: 0.8000
Epoch 17/20
2/2 [==============================] - 0s 38ms/step - loss: 0.5715 - accuracy: 0.8250 - val_loss: 0.5616 - val_accuracy: 0.8750
Epoch 18/20
2/2 [==============================] - 0s 39ms/step - loss: 0.5606 - accuracy: 0.8750 - val_loss: 0.5502 - val_accuracy: 0.8750
Epoch 19/20
2/2 [==============================] - 0s 38ms/step - loss: 0.5491 - accuracy: 0.8750 - val_loss: 0.5381 - val_accuracy: 0.9250
Epoch 20/20
2/2 [==============================] - 0s 38ms/step - loss: 0.5374 - accuracy: 0.9250 - val_loss: 0.5256 - val_accuracy: 0.9250
````
## Contributors
I  would like to thank my buddy [Mridul Rao aka Mani](https://github.com/mridulrao) for helping me out 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.<br />

Please make sure to update tests as appropriate.
