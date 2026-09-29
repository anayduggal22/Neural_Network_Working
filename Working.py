import numpy as np

np.random.seed(13)

# Creating a tiny dataset [XOR Operation]

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Creating weights and biases

# First Layer ( 2 inputs states going to 2 hidden states)

w1 = np.random.randn(2,2)*0.5 
b1 = np.zeros((1,2)) 

# w1 = [[w11, w12],
#      [w21, w22]]

# Second Layer (2 hidden states going to 1 output state)

w2 = np.random.randn(2,1)*0.5 
b2 = np.zeros((1,1)) 

#w2 = [[w21],
#      [w22]]


def sigmoid(x):
    
    return 1 /(1 + np.exp(-x))

def sigmoid_derivative(x):
    
    return x*(1-x)


lr =0.5 # learing rate
epoch= 6000 # epochs

for e in range(epoch):
    
    # FORWARD PASS
    z1 = X@w1 + b1 # First Layer
    h = sigmoid(z1) # Creating h1, and h2
    
    z2 = h@w2 + b2 # Second Layer
    o = sigmoid(z2) # Creating predicted output
    
    
    # Loss Function
    loss = np.mean((y-o)**2)
    
    
    #BACKWARD PASS, using chain rule
    
    # o and z2
    d_loss_o = -2* (y-o)
    d_o_z2 = sigmoid_derivative(o)
    
    d_z2 = d_loss_o * d_o_z2
    
    # w2 and b2
    dw2 = h.T @ d_z2
    db2 = np.sum(d_z2, axis=0, keepdims=True)
    
    # h and z1
    d_h = d_z2 @ w2.T
    d_h_z1 = sigmoid_derivative(h)
    
    d_z1 = d_h * d_h_z1
    
    # w1 and b1
    dw1 = X.T @ d_z1
    db1 = np.sum(d_z1, axis = 0, keepdims=True)
    
    # Updating values using gradient descent
    
    w2 = w2 - lr*dw2
    b2 = b2 - lr*db2
    w1 = w1 - lr*dw1
    b1 = b1 - lr*db1
    

    if e % 2000 == 0:
        print(f"Epoch {e}, Loss: {loss:.4f}")



# Final Predictions

print("\nFinal Predictions:")
print(o.round(3),"\n")
print("True Labels:")
print(y)