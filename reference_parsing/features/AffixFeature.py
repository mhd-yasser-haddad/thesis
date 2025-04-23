class AffixFeature:
    def __init__(self, size=4, prefix=True, suffix=False):
        self.size = size
        self.suffix = suffix or not prefix

    def observe(self, token, **opts):
        chars = self.extract(token)
        return self.build(chars, self.join)

    def extract(self, token):
        if self.suffix:
            return list(token)[-self.size:][::-1]
        return list(token)[:self.size]

    def join(self, chars):
        if self.suffix:
            return ''.join(chars[::-1])
        return ''.join(chars)

    def build(self, chars, join_func):
        return [join_func(chars[:n]) for n in range(1, self.size + 1)]

    def is_suffix(self):
        return self.suffix

    def is_prefix(self):
        return not self.suffix