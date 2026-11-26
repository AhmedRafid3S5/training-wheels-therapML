from pathlib import Path
from therapml.tokenizer import BasicTokenizer
from therapml.tokenizer_analysis import load_corpus, SPECIAL_TOKENS

#load corpus
train_text , _= load_corpus()

#Instantiate and train the tokenizer model on the entire corpus
tokenizer = BasicTokenizer()
tokenizer.train(train_text,vocab_size=10000,verbose=True,special_tokens=SPECIAL_TOKENS)

#Save the model 
tokenizer.save("bpe_10_000")
print("SAVED")