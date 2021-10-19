import camb
import numpy as np
import matplotlib.pyplot as plt

##Import data
planck = np.loadtxt('./COM_PowerSpect_CMB-TT-full_R3.01.txt')
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])

###Problem 1

def get_spectrum(pars,lmax=3000):
    #Get parameter
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    #Get power spectrum
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]
    return tt[2:]

#Define two arrays for the two different sets of parameters we are comparing
pars=np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
pars2 = np.asarray([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])

#Get the chi squared values for both sets of parameters
model=get_spectrum(pars)
model2=get_spectrum(pars2)

model=model[:len(spec)]
model2=model2[:len(spec)]

resid=spec-model
resid2=spec-model2

chisq=np.sum((resid/errs)**2)
chisq2=np.sum((resid2/errs)**2)

#Output the chi squared values calculated above
print("chisq is ",chisq," for parameters",pars, " and ",len(resid)-len(pars)," degrees of freedom.")
print("chisq is ",chisq2," for parameters",pars2, " and ",len(resid2)-len(pars2)," degrees of freedom.")

###Problem 2

#Define a function to take the derivative of get_spectrum with respect to one parameter.
#The other parameters remain fixed. 
def ndiff(fun, args, pos):
    #Get the parameter value at index pos in the list args.
    x = args[pos]
    #Fix a dx of 10^-5 * x.
    dx = 1e-5 * x
    #Update the parameter in the list agrs.
    args[pos] = x+dx
    fx1 = fun(args)
    args[pos] = x-dx
    fx2 = fun(args)
    #Calculate and output the derivative.
    deriv = (fx1 - fx2)/(2*dx)
    return deriv

#Define a function that takes the derivative of get_spectrum with respect to one parameter, for all the parameters.
#The output is a matrix with n rows and m columns where . 
def Deriv(fun, args):
    #Get the size of the output of the function we are trying to differentiate
    y = fun(args)
    #Create the derivative matrix.
    derivs = np.zeros([len(y), len(args)])
    #For each parameter, get its derivative using nidff and update the corresponding ___ in the derivative matrix.
    for i in range(len(args)):
        derivs[:, i] = ndiff(get_spectrum, args, i)
    return derivs

#Create a function to update the lamda parameter used in the LM method. 
def update_lamda(lamda, success):
    if success: 
        lamda = lamda/1.5
        if lamda<0.5:
            lamda=0
    else : 
        if lamda==0:
            lamda=1
        else : 
            lamda=lamda*1.5**2
    return lamda
            
def fit_LM(fun, args, y, error, niter=50, chitol=0.01):
    #Start with a high lambda value.
    lamda = 1
    #Get the initial value of chi squared to start the loop.
    model = fun(args)
    model = model[:len(y)]
    r = y-model
    chisq_old = np.sum((r/error)**2)
    for i in range(niter):
        
        #Calculate the step from the derivative matrix.
        deriv = Deriv(fun, args)
        model = model[:len(y)]
        deriv = deriv[:len(y)]
        r = y-model
        lhs = deriv.T@deriv
        lhs = lhs + lamda*np.diag(np.diag(lhs))
        rhs = deriv.T@r
        dm = np.linalg.inv(lhs)@rhs
        args_trial = args+dm
        
        #Calculate the chi-squared value for the trial step
        model = fun(args_trial)
        model = model[:len(y)]
        r = y-model
        chisq_new = np.sum((r/error)**2)
        
        #Update the current parameters if the trial step improves the model, if not keep iterating.
        if chisq_new<chisq_old:
            lamda = update_lamda(lamda, True)
            args = args_trial
            print('accepting step with new chisq ',chisq_new, 'and old chisq ',chisq_old)
            chisq_old = chisq_new
        elif np.abs(chisq_new-chisq_old)<chitol:
            break
        else :
            lamda = update_lamda(lamda, False)
            print('rejecting step with new chisq ',chisq_new, 'and old chisq ',chisq_old)
        print('On iteration', i, 'chisq is ', chisq_new, 'with step ', dm, 'and lamda ', lamda)
    
    #After getting the best-fit values using the LM method, we calculate the error on our parameters
    #Using the covariance matrix. We also output the curvature matrix for the next problem.
    last_deriv = Deriv(fun, args)
    last_lhs = last_deriv.T@last_deriv
    Cov = np.linalg.inv(last_lhs)
    errors = np.sqrt(np.diag(Cov))
    return args, errors, last_lhs

#Get the best-fit values and errors on our parameters,as well as the curvature matrix. 
Bestfit_args, Bestfit_err, Cur = fit_LM(get_spectrum, pars, spec, errs)
Cur = np.linalg.inv(Cur)

#Write results to a txt file.
line = ['The best-fit parameters are: ','H0: '+str(Bestfit_args[0])+' +/- '+str(Bestfit_err[0]),
        'ombh2: '+str(Bestfit_args[1])+' +/- '+str(Bestfit_err[1]), 
        'omch2: '+str(Bestfit_args[2])+' +/- '+str(Bestfit_err[2]),
        'tau: '+str(Bestfit_args[3])+' +/- '+str(Bestfit_err[3]),
        'As: '+str(Bestfit_args[4])+' +/- '+str(Bestfit_err[4]),
        'ns: '+str(Bestfit_args[5])+' +/- '+str(Bestfit_err[5])]
L = open('./planck_fit_params.txt', 'w')
for elem in line: 
    L.write(elem+'\n')
L.close()



###Problem 3


#Define a function to calculate the mean value of the dark energy using the H0,
# ombh2 and omch2 values generated by the MCMC function.
def Mean_Dark_Energy(H0, ombh2, omch2, H0_err, ombh2_err, omch2_err):
    #We propagate the errors through the various expressions to get a 
    #final value for the mean value of dark energy and the error on this value.
    h = H0 / 100
    h_err = H0_err / 10
    omb = ombh2 / h**2
    omb_err = (omb*h_err)/(h**3)
    omc = omch2 / h**2
    omc_err = (omc*h_err)/(h**3)
    omD = 1 - omb - omc
    omD_err = np.sqrt(omb_err**2 + omc_err**2)
    return omD, omD_err

#Define a function to calculate the chi squared value for a given set of arguments and a set of data points.
#We fix the errors to the ones found in the COM_PowerSpect_CMB-TT-full_R3.01.txt file, since we won't be using
#any other data for these questions. 
def get_spectrum_chisq(args, y, errors=errs):
    model = get_spectrum(args)
    model = model[:len(y)]
    r = y - model
    chisq = np.sum((r/errors)**2)
    return chisq

#Make an MCMC sampler.
def mcmc(pars,y,fun,curv,nstep):
    #We calculate the current value of chi squared to start off the chain.
    chi_cur=fun(pars,y)
    npar=len(pars)
    
    #Initialize the chains for the parameters and chi squared values.
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    #Using the cholesky decomposition we get a single value from the curvature matrix
    #which we can use as our step size. 
    L = np.linalg.cholesky(curv)
    curv = np.dot(L,np.random.randn(npar))
    
    for i in range(nstep):
        #Get the trial parameters that we test out.
        #We add a random value to the step size so as to not get stuck in a bad part of parameter space. 
        trial_pars=pars + 2*curv*np.random.randn(npar)
        
        #Get the chi squared values for the trial parameters.
        trial_chisq=fun(trial_pars,y)
        
        #Get the difference of the old chi squared and the trial chi squared.
        delta_chisq=trial_chisq-chi_cur
        
        #If our chi squared has decreased, we update the pars and chi_cur variables, 
        #as well as the parameter and chi squared chains. 
        if delta_chisq<0:
            pars=trial_pars
            chi_cur=trial_chisq
            chain[i,:]=pars
            chivec[i]=chi_cur
        
        #If our chi squared hasn't decreased, we don't update the pars and chi_cur variables,
        #however we update the two chains, to show that the step was unsuccesful. 
        else:
            chain[i,:]=pars
            chivec[i]=chi_cur
    return chain,chivec

#Get the parameter and chi squared chains.
chain,chisq=mcmc(pars,spec,get_spectrum_chisq,Cur,nstep=500)
print('param chain is: ',chain, 'chi squared chain is: ',chisq)

#Get the mean value of dark energy from the chains
error = np.std(chain, axis=0)
bestfit_MCMC_param = chain[len(chain)-1]
omd, omd_err = Mean_Dark_Energy(bestfit_MCMC_param[0], bestfit_MCMC_param[1], bestfit_MCMC_param[2], error[0], error[1], error[2])
print('The mean value of dark energy is ', omd, ' +/- ',omd_err)
#Write mean value of dark energy to a .txt file.

line = ['The mean value of dark energy is: ', str(omd)+' +/- '+str(omd_err)]
L = open('./Problem3.txt', 'w')
for elem in line: 
    L.write(elem+'\n')
L.close()

#To test the convergence of our chains, we plot the evolution
#of chi squared to make sure that the value converges, as well as the 
#fourier transform of the parameter chain. 

plt.plot(chisq)
plt.xlabel('Iterations')
plt.ylabel('Chi squared')
plt.title('Plot of the evolution of the chisq values')
plt.savefig('./Chisq_Values.pdf')
plt.show()

chainft = np.fft.rfft(chain)
plt.loglog(np.abs(chainft))
plt.xlabel('Iterations (log scale)')
plt.ylabel('Fourier Transform (log scale)')
plt.title('Plot of the evolution of the Fourier Transform of the chain')
plt.savefig('./Convergence.pdf')
plt.show()

#We save the output of both chains to a text file, where the first column is the chi squared value for 
#each iteration of parameters. 
npar=len(bestfit_MCMC_param)
#Create a matrix containing both chains.
line = np.zeros([len(chisq), npar+1])
line[:, 0] = chisq
line[:, 1] = chain[:, 0]
line[:, 2] = chain[:, 1]
line[:, 3] = chain[:, 2]
line[:, 4] = chain[:, 3]
line[:, 5] = chain[:, 4]
line[:, 6] = chain[:, 5]
line=np.matrix(line)
#Write the matrix to a .txt file
with open('./planck_chain.txt','w') as L:
    for elem in line:
        np.savetxt(L, elem, fmt='%.17f')
L.close()

###Problem 4


#To impose a prior on the tau value we use the same mcmc function as in problem 3
#but we sample the value of tau at each iteration from a gaussian distribution with 
#mean 0.0540 and standard deviation 0.0074.

#We re-estimate the covariance matrix from the chains obtained in problem 3.
new_deriv = Deriv(get_spectrum, bestfit_MCMC_param)
new_lhs = new_deriv.T@new_deriv
new_Cov = np.linalg.inv(new_lhs)

def mcmc_withprior(pars,y,fun,curv,nstep):
    chi_cur=fun(pars,y)
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    L = np.linalg.cholesky(curv)
    curv = np.dot(L,np.random.randn(npar))
    for i in range(nstep):
        trial_pars=pars + curv*np.random.randn(npar)
        
        #We sample the tau value for the gaussian distribution.
        trial_pars[3]=np.random.normal(0.0540, 0.0074)
        trial_chisq=fun(trial_pars,y)
        delta_chisq=trial_chisq-chi_cur
        if delta_chisq<0:
            pars=trial_pars
            chi_cur=trial_chisq
            chain[i,:]=pars
            chivec[i]=chi_cur
        else:
            chain[i,:]=pars
            chivec[i]=chi_cur
    return chain,chivec

prior_chain, prior_chisq=mcmc_withprior(pars,spec,get_spectrum_chisq,new_Cov,nstep=500)
print('param chain is: ',prior_chain, 'chi squared chain is: ',prior_chisq)
plt.plot(prior_chisq)

#Save results from mcmc_withprior to a .txt file, using the same method as problem 3
newline = np.zeros([len(prior_chisq), npar+1])
newline[:, 0] = prior_chisq
newline[:, 1] = prior_chain[:, 0]
newline[:, 2] = prior_chain[:, 1]
newline[:, 3] = prior_chain[:, 2]
newline[:, 4] = prior_chain[:, 3]
newline[:, 5] = prior_chain[:, 4]
newline[:, 6] = prior_chain[:, 5]
newline=np.matrix(newline)
#Write the matrix to a .txt file
with open('./planck_chain_tauprior.txt','w') as L:
    for elem in newline:
        np.savetxt(L, elem, fmt='%.17f')
L.close()

#Getting the mean value of each parameter from importance sampling
def prior_chisq(pars, pars_priors, par_errs):
    if pars_priors is None:
        return 0
    par_shifts=pars - pars_priors
    return np.sum((par_shifts/par_errs)**2)

expec_pars = 0*pars
expec_pars[3]=0.0540
par_errs=0*pars+1e20
par_errs[3]=0.0074

nsample=chain.shape[0]
imp_chivec = np.zeros(nsample)
for i in range(nsample):
    chisq = prior_chisq(chain[i,:],expec_pars, par_errs)
    imp_chivec[i]=chisq
imp_chivec = imp_chivec-imp_chivec.mean()
weight=np.exp(0.5*imp_chivec)

imp_pars=np.zeros(len(pars))
for i in range(len(pars)):
    imp_pars[i]=np.sum(weight*chain[:,i])/np.sum(weight)
    
#Comparing importance sampled results to results obtained from an mcmc sampler with prior

prob3_chisq = get_spectrum_chisq(imp_pars,spec)
prob4_chisq = get_spectrum_chisq(prior_chain[len(prior_chain)-1],spec)
print('The chi squared value using an mcmc sampler with importance sampling is:', prob3_chisq)
print('The chi squared value using an mcmc sampler with a prior is:', prob4_chisq)

line = ['The chi squared value using an mcmc sampler with importance sampling is: '+str(prob3_chisq),
        'The chi squared value using an mcmc sampler with a prior is: '+str(prob4_chisq)]
L = open('./Problem4.txt', 'w')
for elem in line: 
    L.write(elem+'\n')
L.close()

###I was unable to run longer MCMC chains due to some technical issues, mostly because I had to switch from a Mac
###to a Windows computer over the weekend.
