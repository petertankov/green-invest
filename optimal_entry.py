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

    
def ls_entry(T0,pi,P,rev,beta,K,desc):
    # Pricing the real option via Longstaff-Schwarz algorithm
    # T0: real option maturity
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
    V = np.zeros((Nsim,T0+1))
    E = np.zeros((Nsim,T0+1))
    E[:,T0] = regress2(rev[:,T0],np.transpose(np.concatenate((pi[:-1,:,T0],P[:,:,T0]),axis=0)),2)
    V[:,T0] = (E[:,T0]-K[T0])*(E[:,T0]>K[T0])
    tau = (T0+1)*np.ones(Nsim)
    tau[E[:,T0]>=K[T0]] = T0
    f = IntProgress(min=0, max=T,description="LS "+desc) # instantiate the bar
    display(f) 
    for t in range(T0-1,0,-1):
        f.value+=1
        E[:,t] = regress2(rev[:,t],np.transpose(np.concatenate((pi[:-1,:,t],P[:,:,t]),axis=0)),2)
        CV = regress2(beta*V[:,t+1],np.transpose(np.concatenate((pi[:-1,:,t],P[:,:,t]),axis=0)),2)
        V[:,t] = E[:,t]-K[t]
        V[E[:,t]-K[t]<CV,t] = CV[E[:,t]-K[t]<CV]
        tau[E[:,t]-K[t]>=CV] = t
    CV = beta*np.mean(V[:,1])
    E0 = np.mean(rev[:,0])
    if(CV<E0-K[0]):
        price = E0 - K[0]
    else:
        price = CV
    al = 0.05
    SE = np.std(beta*V[:,1])/np.sqrt(Nsim)
    ub = price + norm.ppf(1-al/2)*SE 
    lb = price - norm.ppf(1-al/2)*SE
    mtau = np.mean(tau)
    stau = np.std(tau)/np.sqrt(Nsim)
    ltau = mtau-norm.ppf(1-al/2)*stau
    utau = mtau+norm.ppf(1-al/2)*stau
    f.close()
    return price, ub, lb, mtau, utau, ltau, tau

def calctaunan(tau,T0):
    # replaces all elements of tau larger than T0 with NaN and computes the
    # expected stopping time excluding NaNs, the confidence interval and the probability of investment
    tau[tau>T0] = float('nan')
    al = 0.05
    mtau = np.nanmean(tau,axis=1)
    stau = np.nanstd(tau,axis=1)/np.sqrt(np.sum(~np.isnan(tau),axis=1))
    ltau = mtau-norm.ppf(1-al/2)*stau
    utau = mtau+norm.ppf(1-al/2)*stau
    ptau = np.sum(~np.isnan(tau),axis=1)/tau.shape[1]
    return mtau,ltau,utau,ptau

def simulate_entry(ngfs_vars,SS,signal,Nsim,rseed,r,pi0,Phi,Sigma,sigma_fact,sigma_sig,W,R_U,R_C,C_F,C_V):
    beta = np.exp(-r)
    scenarios =  ['Below 2°C', 'Current Policies', 'Delayed transition', 
                 'Divergent Net Zero', 'NDCs', 'Net Zero 2050']
    snames = [scenarios[i] for i in SS]
    print("Selected scenarios: ",snames)
    T_years = 80
    T0_years = 30
    Life_years = 50
    T_months = T_years*12
    months = np.linspace(0,T_months,T_months+1)
    years = np.linspace(0,T_years,T_years+1)
    NGFS_years = np.arange(0,T_years+1,5)

    mu_sig = np.zeros((len(SS),T_months+1))
    R_U_long = R_U*np.ones((len(SS),T_years+1))# Constant use rate
    mu = np.zeros((2,len(SS),T_years+1))

    for i in range(len(SS)):
        if(signal=='P_CO2'):
            mu_sig[i,:] = np.log(np.interp(months/12,NGFS_years , ngfs_vars["P_CO2"][SS[i],:]))
        else:
            mu_sig[i,:] = np.interp(months/12,NGFS_years , ngfs_vars["emissions"][SS[i],:])
        mu[0,i,:] = np.log(np.interp(years,NGFS_years  , ngfs_vars["P_E"][SS[i],:]))
        mu[1,i,:] = np.log(np.interp(years,NGFS_years  , ngfs_vars["P_C"][SS[i],:]))

    CC_C = np.interp(years,NGFS_years,ngfs_vars["CC_C"])

    HPY = 365.25*24 
    K = CC_C*W # RO strike : fraction of capital costs of building the plant , length: T_RO

    revenues = lambda W,E,F,R_C,C_F,R_U,C_V: W*(HPY*R_U*( E - F/R_C  - C_V)- C_F)


    price = np.zeros(len(sigma_fact))
    ub = np.zeros(len(sigma_fact))
    lb = np.zeros(len(sigma_fact))
    mtau = np.zeros(len(sigma_fact))
    utau = np.zeros(len(sigma_fact))
    ltau = np.zeros(len(sigma_fact))
    tau = np.zeros((len(sigma_fact),Nsim))
    mrev = np.zeros((len(sigma_fact),T0_years+1))

    sub_ind = np.arange(0,T_months+1,12,dtype=int)

    start_time = time.time()


    for i in range(len(sigma_fact)):
        pi,lP,I = simTraj(Nsim,pi0,Sigma*sigma_fact[i],Phi,sigma_sig[i],mu_sig,sub_ind,rseed,str(i))

        P = np.zeros(lP.shape)
        for rf in range(2):
            for t in range(T_years+1):
                P[rf,:,t] = np.exp(lP[rf,:,t]+mu[rf,I,t])

        rev = np.zeros((Nsim,T0_years+1))
        f = IntProgress(min=0, max=T_years+1,description="Revenues "+str(i)) # instantiate the bar
        display(f) 
        for t in range(T0_years+1):
            f.value+=1
            for s in range(Life_years):
                rev[:,t] = rev[:,t]+np.power(beta[i],s+1)*revenues(W,P[0,:,t+s+1],P[1,:,t+s+1],R_C,C_F,R_U_long[I,t+s+1],C_V)
        f.close()
        price[i], ub[i], lb[i], mtau[i], utau[i], ltau[i], tau[i,:] = ls_entry(T0_years,pi,lP,rev,beta[i],K,str(i))
    

    calc_time = time.time()

    print('Calculation time: ',calc_time-start_time)
    return price, ub, lb, mtau, utau, ltau, tau, I

