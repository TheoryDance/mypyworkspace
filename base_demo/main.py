# 简写语法学习及lambda

# if简写
x = -1
a = '正数' if x>0 else '负数'
print(a)

# for简写
x = range(10)
c = [i for i in x if i%2==0]
print(c)

# lambda
func = lambda x:x**2
print(func(10))


foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]



# filter
x = list(filter(lambda x:x%3==0,foo))
print(x)
x = [i for i in foo if i%3==0]
print(x)

#map
x = list(map(lambda x:x*2+10, foo))
print(x)
y = [i*2+10 for i in foo]
print('y = ',y)

from functools import reduce
#reduce
x = reduce(lambda x,y:x+y, foo)
print(x)
x = reduce(lambda x,y: ((x>y) and x or y), foo)
print(x)
