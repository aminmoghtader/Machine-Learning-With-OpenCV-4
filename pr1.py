import numpy as np 
import gc
import os



a1 = [1,2,3,4,6]
a2 = [2,3,5,6,9]
a3 = [5,6,9,8,7]
a4 = np.array([a1,a2,a3])
a5 = a4[:2, 1:5]
a4[0][0] = 20
print(a4)
a6 = a5[1,2] = 10
#print(a4)
#print(a4.reshape(-1,))
x = np.array([2,3])
y = np.array([4,2])
z = x * y
#print(z)
#print(np.dot(x,y))
x1 = np.matrix([[1,2],[4,5]])
y1 = np.matrix([[3,5],[8,6]])
#print(x1 * y1)
a = np.array([(1,2,3),(4,5,6),(7,8,9)])
#print(a.cumsum(axis=1))
#print(a.shape)
print(np.sort(a3))
x2 = np.array(a3)
print(x2[x2.argsort()])

person = np.array(['jon','am','ma','will','mag'])
age = np.array([34,12,37,5,13])
heights = np.array([1.76,1.2,1.68,0.5,1.25])

sor = np.argsort(age)[::-1] # برای معکوس کردن
print(person[sor])
print(age[sor])
print(heights[sor])

a10 = a4.copy()
a4[0][1] = 100
a4.shape = 1,-1
print(a4)
print(a10)

gc.collect()
globals().clear()
locals().clear()
