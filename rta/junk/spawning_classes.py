class T(object):
    def __init__(self, a = 10):
        self.a = a

    def __repr__(self):
        return str(self.a)

    def spawn(self):
        return T(a=5)

t = T()
s = t.spawn()
s