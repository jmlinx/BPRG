import numpy as np
from .kernel import KernelMesh

class BPRG():
    def __init__(self, nv, S, L, K, κ, η):
        assert nv % L == 0
        self.nv = nv
        S_lin = np.linspace(S[0], S[1], L)
        self.sv = np.repeat(S_lin, nv/L)
        self.κ = KernelMesh(κ)
        self.L = L
        self.K = K
        self.η = η
        self.result = None
        
    def get_A(self, κ, sv):
        P = κ(sv, sv).numpy() / self.nv # connection prob
        U = np.random.uniform(size=P.shape)
        A = (U <= P)*1 # adjacency matrix
        return A
        
        
    def get_kv(self, nv, η):
        idx = np.arange(nv)
        np.random.shuffle(idx)
        nk_list = (nv * np.array(η)).astype(int)
        nk_idx = nk_list.cumsum()

        kv = np.zeros(nv)
        for nk in nk_idx:
            kv[idx[nk:]] +=1
        return kv
    
    def simulate(self):
        A = self.get_A(self.κ, self.sv)
        kv = self.get_kv(self.nv, self.η)
        
        kv0_checked = np.zeros(self.nv) # explored infection
        kv0_new = (kv==0)*1 # new infection
        
        kv_list = [kv]
        kv0_checked_list = [kv0_checked]
        kv0_new_list = [kv0_new]
        
        while True:
            # track change of remaining threshold
            kv_new = np.clip(kv - kv0_new @ A, 0, None)
            # note checked infection
            kv0_checked += kv0_new 
            if np.equal(kv_new, kv).all():
                break
            else:
                kv = kv_new           
                kv0_new = (kv_new==0)*1 - kv0_checked# drop previously explored 0
            kv_list.append(kv_new)
            kv0_checked_list.append(kv0_checked)
            kv0_new_list.append(kv0_new)
        
        result = {'sv': self.sv,
               'A': A,
               'kv_list': kv_list,
               'kv0_checked_list': kv0_checked_list,
               'kv0_new_list': kv0_new_list,
              }
        
        return result
    
    def cal_infection_pct(self, kv_list):
        return [(kv==0).mean() for kv in kv_list]
    
    def cal_infection_by_type(self, kv):
        kv_L = kv.reshape(self.L, int(self.nv/self.L))
        u0_L = (kv_L==0).sum(axis=1)
        return u0_L
    
    def cal_infection_scale_by_type(self, kv):
        u0_L = self.cal_infection_by_type(kv)
        f_L = u0_L * self.L / self.nv
        return f_L