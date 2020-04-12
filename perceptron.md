

# Perceptron: Overview
***
## The Structure of a Perceptron
An input layer fully connected with an output layer: 

- The input layer contains nodes representative of each attribute in the dataset. Think of each node matching one column in the dataset. *A common cause for confusion in my experience is to visualize the perceptron taking in more than one instance from the dataset at once.* That is not the case; the input layer is a placeholder for one instance at a time.


- The output layer for perceptrons is one node. It contains the result of the dot product between the input and the parameters. Because dot products are linear functions, they are not enough to represent nonlinearities in the data. Neural Networks are called Universal Function Approximators because they can approximate any function. To be able to achieve this, non-linearity is introduced with activation functions like Sigmoid, and ReLU.


{% raw %}
$$a' = \sigma(w a + b) \tag{1}$$
{% endraw %}

In equation (1), $a$ is initially the input vector. $a'$ is the vector of all the activations in the next layer. $w$ is the vector of weights belonging to a'. Each node in $a'$ has as many weights as there are nodes in $a$. $b$ is the vector of biases belonging to $a'$. Each node in $a'$ has only one bias. 

**Note a:** A common confusion in my experience is to think of $w$ and $b$ as the parameters of $a$, while in fact they belong to $a'$.

**Note b:** Sometimes the weights and biases are merged in one parameter vector. This can be done by prepending a 1 in the vector of inputs, and prepending the bias in the vector initially containing only the weights. 

## Training a Perceptron
1. Initially, the parameters are generated randomly. A cost function is chosen to evaluate the perceptron's output:
```python
bias = numpy.random.randn(1, 1)
weights = numpy.random.randn(input_length, 1) # for some integer input_length
```  
2. The choice for a cost function depends on the purpose of the perceptron. I'm starting with an example that uses regression, and there won't be an issue with outliers in my data. Hence I'm going to choose Mean Squared Error. Recall that Mean Absolute Error is more suitable for datasets with outliers. Apparently The Huber Loss is a combination of both, but I have yet to investigate. MSE: 

$$C(w,b) \equiv\frac{1}{2n} \sum_x \| y(x) - a\|^2 \tag{2}$$

**Note a:** You will not always see the division by 2 at the beginning, this confused me at first. It turns out that it's convenient when deriving the function. 

**Note b:** A common and wrong expectation to have is that I'm going to use the cost function to train my model. In fact I only use its gradient.
 
3. Another important choice to make concerns the activation function. It also depends on the purpose of the Perceptron. If you want the model predictions to be between 0 and 1, the sigmoid function works. Between -1 and 1, the hyperbolic tangent function works. ReLU is the standard because it is fast to compute as the derivative is 0 for all negative values (0 included), and 1 for all values greater than 0.

**Note a:** Activation functions enable neural networks to be universal function approximators. Without the sigmoid function, equation (1) is linear. Stacking linear functions ends up being just one big linear function, unable to represent nonlinear relationships. Hence the use of activation functions.

**Note b:** It is important to take into consideration the labels when choosing the activation function. Scale the labels of the dataset to match the scale of the predictions. 

**Note c:** Don't use an activation function on the output layer if you are performing regression and want your model to be free in predicting any range of values.

4. Feedforward an instance by computing equation (1).
```python
def sigmoid(self, z):
  return 1.0/(1.0+np.exp(-z))
z = numpy.dot(weights, instance) + bias.squeeze()
a = sigmoid(z)
``` 
**Note a:** When feedforward is used for training, it is necessary to retain the result of all dot products + bias (z) and activations (a). For inference however, no need to retain these values, I'm simply interested in the final output.
<br/><br/>
5. Calculate the gradient of the cost function: Derive the cost function with respect to the activation function, the dot product z, the weights, and the biases. I don't need to use the chain rule in the case of the Perceptron, but I will once I implement a Neural Network. For now these are the derivatives I need:
$$\nabla_a C = (a^L-y) \tag{3}$$
$$\delta^L = \nabla_a C \odot \sigma'(z^L) \tag{4}$$
$$\frac{\partial C}{\partial b} = \delta \tag{5}$$
$$\frac{\partial C}{\partial w} = instance\;\delta \tag{6}$$

In Python:
```python
# Equation 3
def cost_derivative(self, activation, y):
  return (activation-y)

# Equation 4
def sigmoid_prime(self, z):
  sig = sigmoid(z)
  return sig * (1 - sig)
delta = cost_derivative(activation, y) * sigmoid_prime(z)

# Equation 5
nabla_b = delta

# Equation 6
nabla_w = np.dot(delta, instance.reshape(1,-1))
```

6. Backpropagate, then update the weights and bias using:
$$b_{new} \rightarrow b_{old}-\eta \delta \tag{7}$$

$$w_{new} \rightarrow w_{old}-\eta \;instance\;\delta \tag{8}$$

7. Repeat from step 4 using another instance. Gradient Descent stops when the gradient is 0, close to 0 by some defined value, or after the defined maximum number of iterations has been reached.

# Perceptron: Full Implementation
***


```python

```

