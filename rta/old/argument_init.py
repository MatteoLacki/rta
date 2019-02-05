def foo():
    return 4

def goo(a = foo()):
    return a

goo()
goo(10)