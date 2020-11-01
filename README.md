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
## Working
I have created three different notebooks in this project first  i have done the analysis using textblob which is pretty simple and easy you just need basic 
knowledge on how to use panda,textblob and matplot.lib <br />
Now for the model I have first downloaded tweets using 
