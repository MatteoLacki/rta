import multiprocessing as mp


class T(object):
    def __init__(self):
        self.x = np.arange(10000)

    def square(self, a):
        return a**2

    def x_iter(self):
        for a in self.x:
            yield a

    def test(self, cores_no=mp.cpu_count()):
        with mp.Pool(cores_no) as p:
            out = p.map(self.square,
                        self.x_iter())
        return out

t = T()
t.test()
# this works