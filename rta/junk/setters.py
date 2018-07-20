def memoize(f):
    value = None
    def helper(self):
        if value is None:            
            value = f(x)
        return value
    return helper

class T(object):
    def __init__(self, dupa):
        self._dupa = dupa

    @property
    def dupa(self):
        return self._dupa
    
    @dupa.setter
    def dupa(self, dupa):
        print('Wahaha')
        self._dupa = dupa

    @memoize
    def foo(self):
        print('First call.')
        return 42

t = T(4)
t.dupa
t.dupa = 5
t.foo()