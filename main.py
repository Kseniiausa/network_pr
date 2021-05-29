import numpy as np
# -*- coding: utf-8 -*-
#сайт https://habr.com/ru/post/271563/
# Сигмоида ограничена двумя горизонтальными асимптотами y=1
# что дает нормализацию выходного значения каждого нейрона.
def nonlin(x, der=False):
    if der==True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

#входные данные
x = np.array([[1, 0, 1],
              [1, 0, 1],
              [0, 1, 0],
              [0, 1, 0]])
#вых или ожид исп фунцию переноса
y = np.array([[0, 0, 1, 1]]).T

#случайное распределение будет каждый раз одним и тем же
np.random.seed(1)

#Матрица весов случайная  со средним 0
syn0 = 2*np.random.random((3, 1))-1

for iter in range(100000000):

    # прямое распространение
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))

    # насколько мы ошиблись?
    l1_error = y - l1

    # перемножим это с наклоном сигмоиды
    # на основе значений в l1
    l1_delta = l1_error * nonlin(l1, True)

    # обновим веса
    syn0 += np.dot(l0.T, l1_delta)

print ("Выходные данные после тренировки:")
print (l1)

print ("Новые выходные данные после тренировки:")
l0_new = np.array([1,1,1])
l1_new = nonlin(np.dot(l0_new, syn0))
print (l1_new)