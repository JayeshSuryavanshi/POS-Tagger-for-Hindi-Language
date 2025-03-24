# !pip install nltk==3.8.1
# !pip install indic-nlp-library
import nltk
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('nonbreaking_prefixes')  
nltk.download('indian')

# Function to parse CoNLL-U formatted files
def parse_conllu(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        for line in file:
            if line.startswith('#') or not line.strip():
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            parts = line.split('\t')
            if len(parts) != 10:
                continue
            word = parts[1]
            pos_tag = parts[3]
            sentence.append((word, pos_tag))
        if sentence:
            sentences.append(sentence)
    return sentences

# Define paths to the dataset files
train_file = 'hi_hdtb-ud-train.conllu'
test_file = 'hi_hdtb-ud-test.conllu'

# Parse the datasets
train_sents = parse_conllu(train_file)
test_sents = parse_conllu(test_file)

# Initialize the default tagger with 'NN' as the fallback tag
default_tagger = DefaultTagger('NN')

# Train the taggers with backoff
unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)
bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)
trigram_tagger = TrigramTagger(train_sents, backoff=bigram_tagger)

# Evaluate the tagger
accuracy = trigram_tagger.accuracy(test_sents)
print(f"Model Accuracy: {accuracy:.2%}")

# Sample sentence
sentence = "दो आदमी आए।"

# Tokenize the sentence
tokens = trivial_tokenize(sentence, lang='hi')

# Tag the tokens
tagged_sentence = trigram_tagger.tag(tokens)

# Display the tagged sentence
print("\nTagged Sentence:")
for word, tag in tagged_sentence:
    print(f"{word}/{tag}", end=' ')