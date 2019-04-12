import nltk
from nltk.corpus import indian
from nltk.tag import tnt
import string


nltk.download('punkt')
nltk.download()

tagged_set = 'hindi.pos'
word_set = indian.sents(tagged_set)
count = 0
for sen in word_set:
    count = count + 1
    sen = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sen]).strip()
    print (sen)
print (count)

train_perc = .9

train_rows = int(train_perc*count)
test_rows = train_rows + 1

print (train_rows, test_rows)

data = indian.tagged_sents(tagged_set)
train_data = data[:train_rows]
test_data = data[test_rows:]


pos_tagger = tnt.TnT()
pos_tagger.train(train_data)
pos_tagger.evaluate(test_data)

word_to_be_tagged = u"३९ गेंदों में दो चौकों और एक छक्के की मदद से ३४ रन बनाने वाले परोरे अंत तक आउट नहीं हुए ।"

tokenized = nltk.word_tokenize(word_to_be_tagged)


print(pos_tagger.tag(tokenized))
