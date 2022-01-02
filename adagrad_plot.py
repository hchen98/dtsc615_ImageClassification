# gradient descent optimization with adagrad for a two-dimensional test function
from math import sqrt
from numpy import arange, asarray
from numpy.random import rand
from numpy.random import seed
import matplotlib.pyplot as plt

# To-Do: (DONE)
# - Modify the func to: x^3 + y^2

# objective function
def objective(x, y):
    # FUNC: x^3 + y^2
    return x**4.0 + y**2.0


# derivative of objective function
def derivative(x, y):
    return asarray([(x ** 3) * 4.0, (y*2.0)])


# gradient descent algorithm with adagrad
def adagrad(objective, derivative, bounds, n_iter, step_size):
    # init the return
    solu_eval = list()
    solu = list()

    # generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # list of the sum square gradients for each variable
    sq_grad_sums = [0.0 for _ in range(bounds.shape[0])]

    # run the gradient descent
    for it in range(n_iter):
        # calculate gradient
        gradient = derivative(solution[0], solution[1])
        
        # update the sum of the squared partial derivatives
        for i in range(gradient.shape[0]): sq_grad_sums[i] += gradient[i]**2.0
        
        # build a solution one variable at a time
        new_solution = list()
        
        for i in range(solution.shape[0]):
            # calculate the step size for this variable
            alpha = step_size / (sqrt(sq_grad_sums[i]) + 1e-8)
            
            # calculate the new position in this variable
            value = solution[i] - alpha * gradient[i]
            
            # store this variable
            new_solution.append(value)
            
        # evaluate candidate point
        solution = asarray(new_solution)
        solution_eval = objective(solution[0], solution[1])
        
        # report progress
        # print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
        solu.append(solution)
        solu_eval.append(solution_eval)

    return solu, solu_eval


# seed the pseudo random number generator
seed(1)

# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])

# define the total iterations
n_iter = 50

###############################################################################
# 1 EPOCH:
# define the step size
step_size = 0.1
# n = range(step_size_list)
# perform the gradient descent search with adagrad
# best, score = adagrad(objective, derivative, bounds, n_iter, step_size)
# print('Done!')
# print('f(%s) = %f' % (best, score))
# print(adagrad(objective, derivative, bounds, n_iter, step_size))
###############################################################################


# Multiple EPOCHES:
# init the data for AdaGrad
LR_MINI = 0.1
LR_MAXI = 0.9
LR_STEP_SIZE = 0.1

# generate a list of LR
step_size_list = arange(LR_MINI, LR_MAXI, LR_STEP_SIZE)
# generate a list of iteration - this will be used when plotting the results
step_list = arange(1, (n_iter + 1), 1)
# a list to store AdaGrad results
adagrad_list = list()

# run the AdaGrad over the list of epoches and the get the results
for item in step_size_list:
    adagrad_list.append(adagrad(objective, derivative, bounds, n_iter, item)[1])


# log the init data 
print("Length of epoch: {}".format(len(step_list)))
print("Length of LR list: {}".format(len(step_size_list)))
print("List of LR: ", end="")
print(", ".join(str(x) for x in step_size_list))


# plot the results
counter = 0
for i in adagrad_list:
    plt.plot(step_list, i, "-x", label=step_size_list[counter])
    counter += 1


# display the plot
plt.title("Optimizing X^4 + y^2 changing the learning rate value")
plt.xlabel("Epochs")
plt.ylabel("Evaluation")
plt.legend()
plt.show()
