import numpy as np
import matplotlib.pyplot as plt

## Iris dataset
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()

# Use pandas' DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_names'] = iris.target_names[iris.target]
print(df)

print("# of records:",len(df))
print("value_counts:")
print(df.target_names.value_counts())

## Training data
# Use petal length, petal width
#x_train = df.loc[:,['petal length (cm)', 'petal width (cm)']].to_numpy() # shape:(150, 2)
#x_train = df.loc[:,['petal length (cm)', 'petal width (cm)']].values # shape:(150, 2)
x_train = df.loc[:,['sepal length (cm)', 'sepal width (cm)']].values # shape:(150, 2)
#x_train = df.loc[:,['petal length (cm)', 'sepal length (cm)']].values # shape:(150, 2)
y_train = np.eye(3)[iris.target] # one-hot representation - shape:(150, 3)

print('x_train=',x_train)
print('y_train=',y_train)

# Plot training data
plt.figure(figsize=(8, 5))
for t in range(3):
    x = x_train[iris.target==t][:,0]
    y = x_train[iris.target==t][:,1]
    cm = [plt.cm.Paired([c]) for c in [0,6,11]]
    plt.scatter(x, y, c=cm[t], edgecolors='k', label=iris.target_names[t])
# label
plt.title('Iris dataset')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)') 
#plt.xlabel('sepal length (cm)')
#plt.ylabel('sepal width (cm)') 
plt.legend()
#plt.show()
plt.savefig('Truth.pdf')



from qcl_classification import QclClassification

# Random number                                                                                                                       
random_seed = 0
np.random.seed(random_seed)

#b Circuit paramter
nqubit = 2
c_depth = 2
num_class = 3

# Instance of QclClassification
qcl = QclClassification(nqubit, c_depth, num_class)

# Training with BFGS
res, theta_init, theta_opt = qcl.fit(x_train, y_train, maxiter=5)


# Plot
h = .05  # step size in the mesh
X = x_train
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# Plot model prediction 
def decision_boundary(X, y, theta, title='(title here)'):
    plt.figure(figsize=(8, 5))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    qcl.set_input_state(np.c_[xx.ravel(), yy.ravel()])
    Z = qcl.pred(theta) # Update model parameter theta
    Z = np.argmax(Z, axis=1)
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    #plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    #print("xx,yy,Z :",len(xx[0]),len(yy[:,0]),len(Z[0]),len(Z[:,0]))

    label_pred = []
    for i in range(len(x_train[:,0])):
        for ix in range(len(xx[0])-1):
            for iy in range(len(yy[:,0])-1):
                if x_train[:,0][i] > xx[0][ix] and x_train[:,0][i] <= xx[0][ix+1] and \
                        x_train[:,1][i] > yy[:,0][iy] and x_train[:,1][i] <= yy[:,0][iy+1]:
                    label_pred.append(Z[iy][ix])
                    
    # Plot also the training points
    #plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=label_pred, edgecolors='k', cmap=plt.cm.Paired)

    # label
    plt.title(title)
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    #plt.xlabel('sepal length (cm)')
    #plt.ylabel('sepal width (cm)')
    #plt.show()
    plt.savefig(title+'.pdf')


# Plot with initial parameter
decision_boundary(x_train, iris.target, theta_init, title='Initial')

# Plot with optimized parameter
decision_boundary(x_train, iris.target, theta_opt, title='Optimized')
