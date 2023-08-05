import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here and remove the following error message 

        q_ini = DiscreteD(x=self.q)

        S = np.zeros(tmax)
        S[0] = q_ini.rand(1)
        for i in range(1,tmax):
            j = int(S[i-1])
            A_trans = DiscreteD(x=self.A[j-1,:])
            S[i] = A_trans.rand(1)
            if (S[i] == (self.nStates)) and self.is_finite:
                return S[:i]
        return S
        #print('Not yet implemented')
    
    def forward(self, prob_distr):
        b = prob_distr
        alpha = np.zeros((np.size(b,0),np.size(b,1)))
        alphatmp = np.zeros((np.size(b,0),np.size(b,1)))
        tmp = np.zeros(np.size(b,0))
        c = np.zeros(np.size(b,1)+1)
        # Initialization
        for j in range(np.size(b,0)):
            alphatmp[j][0] = self.q[j]*b[j][0]
        c[0] = np.sum(alphatmp)
        alpha = alphatmp/c[0] 

        # Forward step 
        for t in range(1,np.size(b,1)):
            for j in range(np.size(b,0)):
                for i in range(np.size(b,0)):
                    tmp[i] = alpha[i][t-1]*self.A[i][j]
                alphatmp[j][t] = b[j][t]*sum(tmp)
            c[t] = np.sum(alphatmp[:,t])
            alpha[:,t] = alphatmp[:,t]/c[t] 

        # Termination
        if (np.size(self.A,0)!=np.size(self.A,1)):
            c[t+1] = np.sum(alpha[:,t]*self.A[:,j+1])   
        return alpha,c

    def backward(self, prob_distr, scale_fact):
        b = prob_distr
        c = scale_fact
        beta = np.zeros((np.size(b,0),np.size(b,1)))
        betahat = np.zeros((np.size(b,0),np.size(b,1)))
        tmp = np.zeros(np.size(b,0))
        T = np.size(b,1)-1

        # Initialization
        if (np.size(self.A,0)!=np.size(self.A,1)): # Finite-duration HMM
            for i in range(np.size(b,0)):
                beta[i][T] = self.A[i][T]
                betahat[i][T] = beta[i][T]/(c[T]*c[T+1])
        else: # Infinite-duration HMM
            for i in range(np.size(b,0)):
                beta[i][T] = 1
                betahat[i][T] = 1/(c[T])

        # Backward step:

        for t in reversed(range(np.size(b,1)-1)):
            for i in range(np.size(b,0)):
                for j in range(np.size(b,0)):
                    tmp[j] = self.A[i][j]*b[j][t+1]*betahat[j][t+1]
                betahat[i][t] = np.sum(tmp)/c[t]
                          
        return betahat
            


        

    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass


    def finiteDuration(self):
        pass
    


    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
