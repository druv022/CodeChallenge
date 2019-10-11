from collections import Counter

class Vocabulary():
    """Vocabulary class to keep account of seen words, converting to index or vice versa based on token, document or documents
    """

    def __init__(self, max_size = None, lower=True, unk_token=True, specials = ('<pad>',)):
        self._max_size = max_size
        self._lower = lower
        self._unk = unk_token
        self._token2id = {token: i for i, token in enumerate(specials)}
        self._id2token = list(specials)
        self._token_count = Counter()

    def __len__(self):
        return len(self._token2id)

    def add_token(self, token):
        token = token.lower() if self._lower else token
        self._token_count.update([token])

    def add_documents(self, docs):
        for sent in docs:
            sent = [i.lower() if self._lower else i for i in sent.split()]
            self._token_count.update(sent)

    def add_document(self, doc):
        sent = [i.lower() if self._lower else i for i in doc.split()]
        self._token_count.update(sent)

    def doc2id(self, doc):
        doc = [i.lower() if self._lower else i for i in doc.split()]
        return [self.token_to_id(token) for token in doc]

    def id2doc(self, ids):
        return [self.id_to_token(idx) for idx in ids]

    def build(self):
        token_freq = self._token_count.most_common(self._max_size)
        idx = len(self.vocab)
        for token, _ in token_freq:
            self._token2id[token] = idx
            self._id2token.append(token)
            idx += 1
        if self._unk:
            unk = '<unk>'
            self._token2id[unk] = idx
            self._id2token.append(unk)

    def token_to_id(self, token):
        token = token.lower() if self._lower else token
        return self._token2id.get(token, len(self._token2id) - 1)

    def id_to_token(self, idx):
        return self._id2token[idx]

    @property
    def vocab(self):
        return self._token2id

    @property
    def reverse_vocab(self):
        return self._id2token

    def token_counter(self):
        return self._token_count

class SentencesIterator():
    """Continuous generator function 
    """
    def __init__(self, generator_function):
        self.generator_function = generator_function
        self.generator = self.generator_function

    def __iter__(self):
        # reset the generator
        self.generator = self.generator_function
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result