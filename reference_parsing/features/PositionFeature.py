class PositionFeature:
    def __init__(self, idx='idx', seq='seq'):
        self.idx = idx
        self.seq = seq

    def observe(self, token, **opts):
        i = opts[self.idx]
        n = len(opts[self.seq])
        if i == 0 and i == n - 1:
            return 'only'
        elif i == 0:
            return 'first'
        elif i == n - 1:
            return 'last'
        else:
            return self.ratio(i, n)

    def ratio(self, x, y, precision=10):
        return f'{round((float(x) / y) * precision) if y > 0 else 0}'
