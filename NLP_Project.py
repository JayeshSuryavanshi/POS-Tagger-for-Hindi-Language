import nltk
from nltk.corpus import indian
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger
from sklearn.model_selection import train_test_split

# Ensure necessary NLTK data is downloaded
nltk.download("indian")
nltk.download("punkt")
nltk.download("punkt_tab")

# Load Hindi tagged sentences from the Indian corpus
tagged_sents = indian.tagged_sents("hindi.pos")

# Split data into training and testing sets
train_sents, test_sents = train_test_split(tagged_sents, test_size=0.1, random_state=42)

# Define a default tagger (assigns 'NN' as a fallback tag)
default_tagger = DefaultTagger("NN")

# Train a unigram tagger (fallback to default tagger)
unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)

# Train a bigram tagger (fallback to unigram tagger)
bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)

# Train a trigram tagger (fallback to bigram tagger)
trigram_tagger = TrigramTagger(train_sents, backoff=bigram_tagger)

# Evaluate the model
accuracy = trigram_tagger.evaluate(test_sents)
print(f"Optimized Model Accuracy: {accuracy:.2%}")

# Tag a new sentence
sentence = "३९ गेंदों में दो चौकों और एक छक्के की मदद से ३४ रन बनाने वाले परोरे अंत तक आउट नहीं हुए ।"
tokens = nltk.word_tokenize(sentence)
tagged_sentence = trigram_tagger.tag(tokens)

print("\nTagged Sentence:")
for word, tag in tagged_sentence:
    print(f"{word}/{tag}", end=" ")
