import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
import time
from ipywidgets import IntProgress
from IPython.display import display



def simTraj(Nsim,pi0,Sigma,Phi, sigma_sig,mu_sig,sub_ind,rseed,desc):
    # Simulate Nsim trajectories of risk factors and scenario probabilities
    # pi0: initial probabilities
    # Sigma : risk factor volatility
    # Phi : risk factor mean reversion
    # sigma_sig: signal volatility
    # mu_sig: signal time-dependent mean
    # sub_ind : subsampling indices (the total length of the simulation
    # is the length of mu_sig, but only values with indices in sub_ind are returned)
    # rseed : random seed; calls with the same rseed return the same trajectories
    # desc : any string that will be printed in the progress bar
    # Return values: pi: simulated probabilities, P: simulated risk factors, I: true scenario for each trajectory
    
    Nrf = Sigma.shape[0]
    T = mu_sig.shape[-1]-1
    NS = len(pi0)
    pi = np.zeros((NS,Nsim,len(sub_ind)))
    P = np.zeros((Nrf,Nsim,len(sub_ind)))
    pi_calc = np.zeros((NS,Nsim))
    P_calc = np.zeros((Nrf,Nsim))
    
    for i in range(NS):
        if 0 in sub_ind:
            pi[i,:,0] = pi0[i]
        pi_calc[i,:] = pi0[i]
    
    rng = np.random.default_rng(seed=rseed)
    
    Inum = np.concatenate((np.zeros(1),np.cumsum(rng.multinomial(Nsim,pi0))))
    I = np.zeros(Nsim,dtype=int)
    for i in range(NS):
        I[int(Inum[i]):int(Inum[i+1])] = i
        
    f = IntProgress(min=0, max=T,description="Simulation "+desc) # instantiate the bar
    display(f)    
    noise = np.transpose(rng.multivariate_normal(np.zeros(Nrf),Sigma,Nsim))
    P_calc[:,:] = noise
    if 0 in sub_ind:
        P[:,:,0] = P_calc[:,:]
    for t in range(1,T+1):
        f.value+=1
        noise = np.transpose(rng.multivariate_normal(np.zeros(Nrf),Sigma,Nsim))
        noise_sig = rng.standard_normal(Nsim)
        P_calc = np.dot(Phi,P_calc) + noise 
        if(sigma_sig>0):
            sig = sigma_sig*noise_sig + mu_sig[I,t]
            for j in range(NS):
                pi_calc[j,:] = pi_calc[j,:]*np.exp(-(sig-mu_sig[j,t])**2/2/sigma_sig**2)
            pi_sum = np.sum(pi_calc[:,:],axis=0)
            for j in range(NS):
                pi_calc[j,:] = pi_calc[j,:]/pi_sum
        else:
            pi_calc[:,:] = 0
            for i in range(NS):
                pi_calc[i,int(Inum[i]):int(Inum[i+1])] = 1
        if t in sub_ind:
            ind = np.nonzero(sub_ind==t)[0][0]
            pi[:,:,ind]=pi_calc
            P[:,:,ind]=P_calc
    f.close()
    return pi,P,I


def regress2(Y,X,degree):
    poly_model = PolynomialFeatures(degree=degree)
    poly_x_values = poly_model.fit_transform(X)
    regression_model = LinearRegression()
    regression_model.fit(poly_x_values, Y)
    return regression_model.predict(poly_x_values)

    
def ls_exit(pi,P,rev,beta,K,desc):
    # Pricing the real option via Longstaff-Schwarz algorithm
    # pi : simulated scenario probabilities
    # P : simulated risk factors
    # rev : simulated revenues
    # beta : discount factors
    # K : capital costs
    # desc : any string that will be printed in the progress bar
    # Return values: 
    # price : option price 
    # ub, lb : upper and lower Monte Carlo bounds for the option price 
    # mtau, utau, ltau : average exit time
    # tau : stopping time values on each simulated scenario
    T = P.shape[2]-1
    Nsim = P.shape[1]
    Vtil = np.zeros((Nsim,T+1))
    V = np.zeros((Nsim,T))
    Vtil[:,T] = rev[:,T] - K[T]
    tau = T*np.ones(Nsim)
    f = IntProgress(min=0, max=T,description="LS "+desc) # instantiate the bar
    display(f) 
    for t in range(T-1,0,-1):
        f.value+=1
        CV = regress2(beta*Vtil[:,t+1],np.transpose(np.concatenate((pi[:-1,:,t],P[:,:,t]),axis=0)),2)
        V[CV+K[t]<0,t] = -K[t]
        tau[CV+K[t]<0] = t
        Vtil[CV+K[t]<0,t] = rev[CV+K[t]<0,t]-K[t]
        V[CV+K[t]>=0,t] = CV[CV+K[t]>=0]
        Vtil[CV+K[t]>=0,t] = rev[CV+K[t]>=0,t]+CV[CV+K[t]>=0]
    V[:,0] = beta*Vtil[:,1]
    price = np.mean(V[:,0])
    al = 0.05
    SE = np.std(V[:,0])/np.sqrt(Nsim)
    ub = price + norm.ppf(1-al/2)*SE 
    lb = price - norm.ppf(1-al/2)*SE
    mtau = np.mean(tau)
    stau = np.std(tau)/np.sqrt(Nsim)
    ltau = mtau-norm.ppf(1-al/2)*stau
    utau = mtau+norm.ppf(1-al/2)*stau
    f.close()
    return price, ub, lb, mtau, utau, ltau, tau


def simulate_exit(ngfs_vars,SS,Nsim,rseed,r,decom_cost,pi0,Phi,Sigma,sig_fact,sigma_sig,W,R_C,C_F,C_V,emission_rate):
    beta = np.exp(-r)
    T_years = 30
    T_months = T_years*12
    months = np.linspace(0,T_months,T_months+1)
    years = np.linspace(0,T_years,T_years+1)
    NGFS_years = np.linspace(0,T_years,7)
    mu_sig = np.zeros((len(SS),T_months+1))
    R_U_long = np.zeros((len(SS),T_years+1))
    mu = np.zeros((3,len(SS),T_years+1))
    scenarios =  ['Below 2°C', 'Current Policies', 'Delayed transition', 
                  'Divergent Net Zero', 'Nationally Determined Contributions (NDCs)', 'Net Zero 2050']
    
    snames = [scenarios[i] for i in SS]
    print("Selected scenarios: ",snames)
    CC_C = np.interp(years,NGFS_years,ngfs_vars["CC"])
    K = decom_cost*CC_C*W
    HPY = 365.25*24


    for i in range(len(SS)):
        mu_sig[i,:] = np.interp(months/12,NGFS_years , ngfs_vars["emissions"][SS[i],:])
        mu[0,i,:] = np.log(np.interp(years,NGFS_years  , ngfs_vars["P_E"][SS[i],:]))
        mu[1,i,:] = np.log(np.interp(years,NGFS_years  , ngfs_vars["P_C"][SS[i],:]))
        mu[2,i,:] = np.log(np.interp(years,NGFS_years  , ngfs_vars["P_CO2"][SS[i],:]))
        R_U_long[i,:] = np.interp(years,NGFS_years,ngfs_vars["R_U"][SS[i],:])
        
    price = np.zeros(len(sig_fact))
    ub = np.zeros(len(sig_fact))
    lb = np.zeros(len(sig_fact))
    mtau = np.zeros(len(sig_fact))
    utau = np.zeros(len(sig_fact))
    ltau = np.zeros(len(sig_fact))
    tau = np.zeros((len(sig_fact),Nsim))

    sub_ind = np.arange(0,T_months+1,12,dtype=int)

    revenues = lambda W,E,F,C,R_C,C_F,R_U,C_V: W*HPY*R_U*( E - F/R_C - emission_rate*C - C_V)- W*C_F 

    
    start_time = time.time()
    for i in range(len(sig_fact)):
        pi,lP,I = simTraj(Nsim,pi0,sig_fact[i]*Sigma,Phi,sigma_sig[i],mu_sig,sub_ind,rseed,str(i))
        P = np.zeros(lP.shape)
        for rf in range(3):
            for t in range(T_years+1):
                P[rf,:,t] = np.exp(lP[rf,:,t]+mu[rf,I,t])

        rev = np.zeros((Nsim,T_years+1))
        f = IntProgress(min=0, max=T_years+1,description="Revenues "+str(i)) # instantiate the bar
        display(f) 
        for t in range(T_years+1):
            f.value+=1
            rev[:,t] = revenues(W,P[0,:,t],P[1,:,t],P[2,:,t],R_C,C_F,R_U_long[I,t],C_V)
        f.close()
        price[i], ub[i], lb[i], mtau[i], utau[i], ltau[i], tau[i,:] = ls_exit(pi,lP,rev,beta[i],K,str(i))
    
    calc_time = time.time()
    print('Calculation time: ',calc_time-start_time)
    return price, ub, lb, mtau, utau, ltau, tau, I
