class Beam:
    def __init__(self, prob=1., data=''):
        self.prob = prob
        self.data = data

    def update(self, prob, word):
        self.prob = prob
        self.data += word + ' '
        return self

    def __str__(self):
        return 'p = {}, data = {}'.format(self.prob, self.data)
