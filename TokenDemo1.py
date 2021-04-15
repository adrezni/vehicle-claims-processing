from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick, text_to_word_sequence

# define 5 documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)  # Basically, creates a word index and other info
# summarize what was learned
print("Word counts:  \n{}".format(t.word_counts))  # Dict of words and their count
                                                    # In this example there are 8 unique words
print("Document count:  \n{}".format(t.document_count)) # Number of docs in the fit
print("Word index:  \n{}".format(t.word_index)) # Dict of words and their indexes. Index starts at 1
                                                # since 0 is reserved.
print("Word docs:  \n{}".format(t.word_docs)) # Dict of words and how many docs each appeared in
# integer encode documents
encoded_docs = t.texts_to_matrix(docs) # Matrix of words of each doc. Uses t.word_index for each doc.
print("Encoded docs matrix:  \n{}".format(encoded_docs))

