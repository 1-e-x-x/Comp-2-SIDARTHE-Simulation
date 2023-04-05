import numpy as np
import matplotlib.pyplot as plt
import csv



sim_history = []

def make_events():
    toReturn = []
    #day 4 introduce basic social-distancing, epidemic awareness
    toReturn.append([4,  {"alpha":0.422, "beta":0.0057, 
                          "delta":0.0057, "gamma":0.285}])
    #day 12 limit screening policy to symptomatic individuals
    toReturn.append([12, {"epsilon":0.143}])
    #day 22 initiate partial lockdown
    toReturn.append([22, {"alpha":0.360, "beta":0.005 , "delta":0.005 , 
                          "gamma":0.200, "zeta":0.034, "eta":0.034, 
                          "mu":0.008, "nu":0.015, "lambd":0.08, 
                          "rho":0.017, "kappa":0.017, "xi":0.017, "sigma":0.017}])
    #day 28 lockdown is operational and gets stricter
    toReturn.append([28, {"alpha":0.210, "gamma":0.110}])
    #day 38 launch wider testing campaign
    toReturn.append([38, {"epsilon":0.200, "rho":0.020, "kappa":0.020, 
                          "xi":0.020, "sigma":0.010, "zeta":0.025, "eta":0.025}])

    return toReturn

class sim():
    def __init__(self, init_state, init_params, init_t, init_dt, events, scheme=True, use_events = True):
        #init state should be a dict of: 
        # {S: value, I: value, D: value, A: value, R: value, T: value, H: value, E: value}
        #init params should be a dict of: 
        # {"alpha": value, "beta": value, "delta": value, "gamma": value, "epsilon": value, "theta": value, "zeta": value, "eta": value, "mu": value, "nu": value, "tau": value, "lambd": value, "rho": value, "kappa": value, "xi": value, "sigma": value}
        #init_t is the intitial time.
        #init_dt is the simulation timestep
        #an event is an array containing a time, and a dict of parameters: [time, params]
        #pass in an array containing every event you want the sim to run: events = [[time1, params1], [time2, params2], etc]
        #set scheme=True for rk4 integration, or scheme=False for euler integration
        #set use_events to True to use events
        self.dt = init_dt
        self.t = init_t
        self.scheme = scheme
        self.state = [init_state["S"], init_state["I"], init_state["D"], init_state["A"], init_state["R"], init_state["T"], init_state["H"], init_state["E"]]
        self.params = init_params
        self.events = events
        self.use_events = use_events

        #add initial state to simulation history
        sim_history.append(self.state)

    def evaluate(self, state):
        #returns state changes
        return np.array([self.dS(state), self.dI(state), self.dD(state), self.dA(state), self.dR(state), self.dT(state), self.dH(state), self.dE(state)])
    
    def euler(self, dt, state, update_function):
        #euler update integration scheme
        return state + dt * update_function(state)

    def rk4(self, dt, state, update_function):
        #runge-kutta integration scheme
        #based off code from here: https://prappleizer.github.io/Tutorials/RK4/RK4_Tutorial.html
        k1 = dt * update_function(state)
        k2 = dt * update_function(state + 0.5*k1)
        k3 = dt * update_function(state + 0.5*k2)
        k4 = dt * update_function(state + k3)
        new_state = state + (1/6)*(k1+2*k2+2*k3+k4)
        return new_state
    
    def check_events(self):
        i=0
        while i < len(self.events):
            if self.events[i][0] <= self.t:
                # print(self.events[i])
                self.run_event(self.events[i][1])
                self.events.pop(i)
            i += 1
    
    def run_event(self, event):
        for keys in event:
            self.params[keys] = event[keys]

    def run(self, time):
        while (self.t < time):
            if(self.use_events):
                self.check_events()
            if(self.scheme):
                self.state = self.rk4(self.dt, self.state, self.evaluate)
            else:
                self.state = self.euler(self.dt, self.state, self.evaluate)
            sim_history.append(self.state)
            self.t += self.dt

    #the differential equations used in the SIDARTHE model
    def dS(self, state):
        return -1 * state[0] * (self.params["alpha"]* state[1] + self.params["beta"]* state[2] + self.params["gamma"] * state[3] + self.params["delta"] * state[4])
    def dI(self, state):
        return state[0]*(self.params["alpha"] * state[1] + self.params["beta"] * state[2] + self.params["gamma"] * state[3]+self.params["delta"]* state[4])-(self.params["epsilon"]+self.params["zeta"]+self.params["lambd"])*state[1]
    def dD(self, state):
        return self.params["epsilon"] * state[1]- (self.params["eta"] + self.params["rho"])* state[2]
    def dA(self, state):
        return self.params["zeta"]* state[1]-(self.params["theta"]+self.params["mu"]+self.params["kappa"])* state[3]
    def dR(self, state):
        return self.params["eta"]* state[2]+self.params["theta"]* state[3]-(self.params["nu"]+self.params["xi"])* state[4]
    def dT(self, state):
        return self.params["mu"]* state[3]+self.params["nu"]* state[4]-(self.params["sigma"]+self.params["tau"])* state[5]
    def dH(self, state):
        return self.params["lambd"]* state[1]+self.params["rho"]* state[2]+self.params["kappa"]* state[3]+self.params["xi"]* state[4]+self.params["sigma"]* state[5]
    def dE(self, state):
        return self.params["tau"]* state[5]

def main():
    #initial state values used by the researchers
    sample_state = {"S": 0.9999962833333332,
                    "I": (200/(60*10**6)),
                    "D": (20/(60*10**6)),
                    "A": (1/(60*10**6)),
                    "R": (2/(60*10**6)),
                    "T": 0,
                    "H": 0,
                    "E": 0}
    #initial parameter values used by the resesearchers
    sample_params = {"alpha": 0.570,
                     "beta": 0.011, 
                     "delta": 0.011,
                     "gamma": 0.456,
                     "epsilon": 0.171,
                     "theta": 0.371,
                     "zeta": 0.125,
                     "eta": 0.125,
                     "mu": 0.017,
                     "nu": 0.027,
                     "tau": 0.01,
                     "lambd": 0.034,
                     "rho": 0.034,
                     "kappa": 0.017,
                     "xi": 0.017,
                     "sigma": 0.017}
    events = make_events()
    end_time = 350 #in days
    step_size = 1 #in days.
    #making the step size too large or too small will break the sim due to the limits of 32bit floating points.
    #the best part of making sims is breaking sims :)

    sampleSim = sim(sample_state, sample_params, 0, step_size, events, True, True)
    
    sampleSim.run(end_time)

    #code to spit out the results of the sim in a .csv file.
    #left in as comments
    # with open('sim_output.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["day","s","i","d","a","r","t","h","e"])
    #     day = 0
    #     for dicts in sim_history:
    #         # writer.writerow([day, dicts["S"], dicts["I"], dicts["D"], dicts["A"], dicts["R"], dicts["T"], dicts["H"], dicts["E"]])
    #         writer.writerow([day, dicts[0], dicts[1], dicts[2], dicts[3], dicts[4], dicts[5], dicts[6], dicts[7]])
    #         day+=step_size
    # file.close()

    s_vals = []
    i_vals = []
    d_vals = []
    a_vals = []
    r_vals = []
    t_vals = []
    h_vals = []
    e_vals = []
    for states in sim_history:
        s_vals.append(states[0])
        i_vals.append(states[1])
        d_vals.append(states[2])
        a_vals.append(states[3])
        r_vals.append(states[4])
        t_vals.append(states[5])
        h_vals.append(states[6])
        e_vals.append(states[7])
    
    n_steps = np.arange(0, len(i_vals))
    print(len(n_steps))
    print(len(i_vals))

    # plt.plot(n_steps, s_vals, label='S')
    plt.plot(n_steps, i_vals, label='I')
    plt.plot(n_steps, d_vals, label='D')
    plt.plot(n_steps, a_vals, label='A')
    plt.plot(n_steps, r_vals, label='R')
    plt.plot(n_steps, t_vals, label='T')
    # plt.plot(n_steps, h_vals, label='H')
    # plt.plot(n_steps, e_vals, label='E')
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Population Proportion")
    plt.title(("With Events, RK4 integration, dt = " + str(step_size)))
    plt.show()



if __name__ =='__main__':
    main()




#Unused code that I'm storing in case I need it later

# self.alpha = init_params["alpha"]
# self.beta = init_params["beta"]
# self.delta = init_params["delta"]
# self.gamma = init_params["gamma"]
# self.epsilon = init_params["epsilon"]
# self.theta = init_params["theta"]
# self.zeta = init_params["zeta"]
# self.eta = init_params["eta"]
# self.mu = init_params["mu"]
# self.nu = init_params["nu"]
# self.tau = init_params["tau"]
# self.lambd = init_params["lambd"]
# self.rho = init_params["rho"]
# self.kappa = init_params["kappa"]
# self.xi = init_params["xi"]
# self.sigma = init_params["sigma"]