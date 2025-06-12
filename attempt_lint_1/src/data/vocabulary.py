import json

class Vocabulary:
    """
    Manages token to ID mappings and vice-versa, handles special tokens.
    """
    def __init__(self, token_list=None, special_tokens=None):
        self.token_to_id_map = {}
        self.id_to_token_map = []
        self.special_tokens = special_tokens if special_tokens else {
            "<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            if token not in self.token_to_id_map:
                self.token_to_id_map[token] = idx
                # Ensure id_to_token_map is large enough
                while len(self.id_to_token_map) <= idx:
                    self.id_to_token_map.append(None)
                self.id_to_token_map[idx] = token

        if token_list:
            self.add_tokens(token_list)

    def add_token(self, token):
        if token not in self.token_to_id_map:
            new_id = len(self.id_to_token_map)
            self.token_to_id_map[token] = new_id
            self.id_to_token_map.append(token)
            return new_id
        return self.token_to_id_map[token]

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)

    def token_to_id(self, token):
        return self.token_to_id_map.get(token, self.special_tokens.get("<unk>"))

    def id_to_token(self, id):
        return self.id_to_token_map[id] if id < len(self.id_to_token_map) else self.special_tokens.get("<unk>")

    def __len__(self):
        return len(self.id_to_token_map)

    @property
    def pad_id(self):
        return self.special_tokens["<pad>"]

    @property
    def sos_id(self):
        return self.special_tokens["<sos>"]

    @property
    def eos_id(self):
        return self.special_tokens["<eos>"]

    @property
    def unk_id(self):
        return self.special_tokens["<unk>"]

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "token_to_id": self.token_to_id_map,
                "id_to_token": self.id_to_token_map,
                "special_tokens": self.special_tokens
            }, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls(special_tokens=data["special_tokens"])
        vocab.token_to_id_map = data["token_to_id"]
        vocab.id_to_token_map = data["id_to_token"]
        return vocab