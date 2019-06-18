from numpy import *
from matplotlib import pyplot as plt

def find_error(m,b,data):
	total_error = 0
	for i in range(0,len(data)):
		x = data[i,0]
		y = data[i,1]
		total_error += (y-(m*x+b))**2
	return total_error/float(len(data))

def step_gradient(current_m,current_b,learning_rate,points):
	m_gradient = 0
	b_gradient = 0
	N = float(len(points))
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		m_gradient += -(2/N) * x * (y-(current_m * x + current_b))
		b_gradient += -(2/N) * (y-(current_m * x + current_b))
	new_m = current_m - (learning_rate*m_gradient)
	new_b = current_b - (learning_rate*b_gradient)
	return [new_m,new_b]

def gradient_runner(starting_m,starting_b,iterations,learning_rate,data):
	m = starting_m
	b = starting_b
	for i in range(iterations):
		m,b = step_gradient(m,b,learning_rate,array(data))
	return [m,b]


def program():
	data = genfromtxt('D:\GitHub\linear_regression_live\data.csv', delimiter=',')
	initial_m = 0
	initial_b = 0
	iterations = 1000
	learning_rate= 0.0001
	print("The Error during m=0, and b=0 is %f"%(find_error(initial_m,initial_b,data)))
	print("Your ML program is Running....")
	m,b = gradient_runner(initial_m,initial_b,iterations,learning_rate,data)
	print("After %d iterations, the m = %d and the b = %d and the error value is %d"%(iterations,m,b,find_error(m,b,data)))
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
