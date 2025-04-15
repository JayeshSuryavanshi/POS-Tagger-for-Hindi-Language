from conllu import parse_incr

def load_conllu_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        sentences, tags = [], []
        for tokenlist in parse_incr(f):
            words = [token['form'] for token in tokenlist if isinstance(token['id'], int)]
            pos_tags = [token['upostag'] for token in tokenlist if isinstance(token['id'], int)]
            sentences.append(words)
            tags.append(pos_tags)
    return sentences, tags
