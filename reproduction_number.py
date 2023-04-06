# Set up initial parameters
alpha = 0.570
beta = 0.011
delta = 0.011
gamma = 0.456
epsilon = 0.171
theta = 0.371
zeta = 0.125
eta = 0.125
mu = 0.017
nu = 0.027
tau = 0.01
lambd = 0.034
rho = 0.034
kappa = 0.017
xi = 0.017
sigma = 0.017

# Calculate r-values
def r1():
    return epsilon + zeta + lambd

def r2():
    return eta + rho

def r3():
    return theta + mu + kappa

def r4():
    return nu + xi

def r5():
    return sigma + tau

# Calculate reproduction rate
def R0():
    p1 = alpha/r1()
    p2 = (beta*epsilon)/(r1()*r2())
    p3 = (gamma*zeta)/(r1()*r3())
    p4 = (delta*eta*epsilon)/(r1()*r2()*r4())
    p5 = (delta*zeta*theta)/(r1()*r3()*r4())

    return p1 + p2 + p3 + p4 + p5


print(R0())