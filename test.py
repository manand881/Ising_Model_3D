import numpy

a=numpy.zeros((3,4,4),dtype=int)
a[1::2,::2,::2] = -1
a[::2,1::2,::2] = -1
a[::2,::2,1::2] = -1
print(a)

def test():
    a=1
    b=2
    return a,b

c,d=test()
print(c,d)