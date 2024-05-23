import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return x**2 - x*0.8 + 0.04

print("Введите интервал в формате \"a,b\":", end = " ")
a0,b0 = [float(x) for x in input().split(",")]
print("Введите точность:", end = " ")
l = float(input())

Z = []
Y = []
Y.append(a0 + (b0-a0)*(3-5**0.5)/2)
Z.append(a0 + b0 - Y[0])
step = 0

A = []
B = []
A.append(a0)
B.append(b0)

while True:
    if f(Y[step]) <= f(Z[step]):
        A.append(A[step])
        B.append(Z[step])
        Y.append(A[step+1] + B[step+1] - Y[step])
        Z.append(Y[step])
    else:
        A.append(Y[step])
        B.append(B[step])
        Y.append(Z[step])
        Z.append(A[step+1] + B[step+1] - Z[step])
    print("Шаг ", step, ":\na = ", A[step + 1], "b = ", B[step + 1], "\nz = ", Z[step + 1], "y = ", Y[step + 1], 
            "\nf(a) =", f(A[step+1]), "f(b) =", f(B[step+1])) 
    print()
    delta = abs(A[step+1] - B[step+1])
    if delta < l:
        AB_ans = [A[step+1], B[step+1]]
        x_ans = (A[step+1] + B[step+1])/2
        break
    else:
        step += 1
    
    if step >= 10000: break
print("Точка минимума x* принадлежит ", AB_ans, ", приближенное значение: ", x_ans)                                                   

plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')
x = np.linspace(a0, b0, 1000)

y = []#x**2 - x*0.4 + 0.04
for xi in x:
    y.append(f(xi))
plt.plot(x, y)

plt.scatter(A, np.array([0]*len(A)), marker='o', color ="red")
plt.scatter(B, np.array([0]*len(A)), marker='o', color ="red")

for i in range(len(A)):
    plt.plot([A[i], A[i]], [max(0,f(A[i])), min(0,f(A[i]))-0.2*i])
    plt.annotate('A' + str(i), [A[i], min(0,f(A[i]))-0.2*i])
    plt.plot([B[i], B[i]], [max(0,f(B[i])), min(0,f(B[i]))-0.2*i])
    plt.annotate('B' + str(i), [B[i], min(0,f(B[i]))-0.2*i])


plt.plot([x_ans, x_ans], [f(x_ans), f(x_ans)-0.2])
plt.annotate('x*', [x_ans, f(x_ans)-0.2])
plt.scatter(x_ans, f(x_ans), marker='o', color ="red")
plt.show()