class SimpleTokenizer:
    def __init__(self, vocab_file):
        self.vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = idx

        self.cls_token_id = self.vocab.get('[CLS]', 101)
        self.sep_token_id = self.vocab.get('[SEP]', 102)
        self.pad_token_id = self.vocab.get('[PAD]', 0)
        self.unk_token_id = self.vocab.get('[UNK]', 100)

    def tokenize(self, text):
        tokens = []
        for word in text.lower().split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                for char in word:
                    tokens.append(self.vocab.get(char, self.unk_token_id))
        return tokens

    def encode(self, text, max_length=64):
        tokens = self.tokenize(text)
        tokens = tokens[:max_length - 2]
        input_ids = [self.cls_token_id] + tokens + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        padding_length = max_length - len(input_ids)
        input_ids      += [self.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask
        }
