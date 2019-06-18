from numpy import *
from matplotlib import pyplot as plt

def find_error(m,b,data):
	totalerror=0
	for i in range(len(data)):
		x = data[i,0]
		y = data[i,1]
		totalerror += ((m*x+b) - y)**2
	return totalerror/float(2*len(data))

def gradient_descent(starting_m,starting_b,learning_rate,points):
	m_gradient = 0
	b_gradient = 0
	N = float(len(points))
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		m_gradient += ((1/N) * x * (starting_m * x + starting_b - y))
		m_gradient += ((1/N)  * (starting_m * x + starting_b - y))
	new_m = starting_m - (learning_rate*m_gradient)
	new_b = starting_b - (learning_rate*b_gradient)
	return [new_m,new_b]

def gradient_runner(starting_m,starting_b,iterations,learning_rate,data):
	m = 0
	b = 0
	for i in range(iterations):
		m,b = gradient_descent(m,b,learning_rate,data)
	return [m,b]


def program():
	data = genfromtxt('D:\GitHub\linear_regression_live\data.csv', delimiter=',')
	initial_m = 0
	initial_b = 0
	learning_rate = 0.0001
	iterations = 1000
	print("At m=0 and b=0, the error is %f"%(find_error(0,0,data)))
	print("Running.....")
	m,b = gradient_runner(initial_m,initial_b,iterations,learning_rate,data)
	print("After %d iterations the m = %f and b = %f and the error value is %f"%(iterations, m,b,find_error(m,b,data)))
	X = []
	Y = []
	for i in range(len(data)):
		X.append(data[i,0])
		Y.append(data[i,1])
	X = array(X)
	Y = array(Y)
	plt.scatter(X, Y)
	plt.plot(X, (m*X+b), color='red')
	plt.show()

program()