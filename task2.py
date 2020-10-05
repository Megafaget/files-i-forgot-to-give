#functions needed
import numpy as np
import pandas as pd  #this one is needed to read data
import matplotlib.pyplot as plt

#file path for panda to read
path = 'C:\Python\AI_work\Annual.csv'
df = pd.read_csv(path)

x = df[['Date']]
y = df['Price']

x = np.array(x)
y = np.array(y)

x = x.reshape(-1,1)
y = y.reshape(-1,1)

def gradient_descent(x,y):
    m_current = b_current = 0
    iterations = 5000
    y_plot = []
    n = float(len(y))
    learning_rate = 0.000000001

    for i in range(iterations):
        y_predicted = m_current * x + b_current
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        debug1 = sum([val**2 for val in (y-y_predicted)])
        y_plot.append(cost)
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_current = m_current - learning_rate * md
        b_current = b_current - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_current,b_current,cost, i))
        #print(y-y_predicted)
    return m_current, b_current, cost, y_plot

m_current, b_current, cost, y_plot = gradient_descent(x,y)

plt.plot(x,y)
#plt.plot(list(range(5000)), y_plot, '-r')
plt.show()
