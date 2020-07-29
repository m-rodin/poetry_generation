## Based on code for "Shall I Compare Thee to a Machine-Written Sonnet? An Algorithmic Approach to Sonnet Generation"

Paper available at: https://arxiv.org/abs/1811.05067

Peter Hase, John Benhart, Tianlin Duan, Liuyi Zhu, Cynthia Rudin

Duke Data Science Team

**Instructions for Producing Sonnets** 

First install the required dependencies (given python 3.6x):

1) pytorch
1) torchtext
2) gensim	
3) numpy	
4) argparse	
5) nltk

Then download the 6 billion tokenn GloVe dictionary from [2] and unzip contents into *glove* directory

Execute the following to train model:
python train.py

Execute the following to generate a poem:
python generate.py *topic seed*

where *topic* is the user-supplied topic of the poem and *seed* (an optional argument) is an integer for the seed.

We require the words in the topic to exist in the 6 billion token GloVe dictionary [2].
Generated poems can be found in the *output_poems* folder. 
