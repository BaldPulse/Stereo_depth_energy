from Energy import *
import numpy as np
import igraph

OCCLUDED = -1
VAR_ALPHA = 1
VAR_ABSENT = 0
MAX_ITER = 4

def IS_VAR(var):
    return var>1 # 0: absent 1: alpha

class Solver:
    def __init__(self, limg, rimg, windowsize, maxdisparity, initial_f=None,  d=2, cutoff=30, K=None):
        self.limg = limg.astype(np.float64)
        self.rimg = rimg.astype(np.float64)
        # windowsize * 2 + 1 = full window length
        self.windowsize = windowsize #windowsize in data term
        # configuration (represented by disparity)
        if initial_f is not None:
            self.d_left = initial_f 
            self.d_right = np.zeros_like(initial_f)
            for i in range(self.limg.shape[0]):
                for j in range(self.limg.shape[1]):
                    p = (i,j)
                    dl = self.d_left[p]
                    if(dl != OCCLUDED):
                        self.d_right[p[0], p[0]+dl] = -dl
        else:
            self.d_left = np.full_like(limg.astype(int), -1)
            self.d_right = np.full_like(rimg.astype(int), -1)
        # disparity matching window along the rectified image rows
        self.maxdisparity = maxdisparity
        # Graph representation of energy function
        self.energy = None
        self.d=d #d-norm in data term
        self.cutoff=cutoff #cutoff value of abs difference in data term
        self.preE = np.inf # previous minimized energy
        # arrays index by pixel p to track the nodes associated to each pixel
        self.varsA = np.zeros_like(self.limg)
        self.vars0 = np.zeros_like(self.limg)

        self.limg_padded = np.pad(self.limg, windowsize, mode="reflect")
        self.rimg_padded = np.pad(self.rimg, windowsize, mode="reflect")

        if K is not None:
            self.K = K
            self.lam = K/5
        else:
            self.tune_K()

    def disparity(self, asn):
        # assignment a - (p, q)
        p, q = asn
        return q[1] - p[1]
    
    def is_active_assignment(self, a):
        return self.f[a[0][0], a[0][1], a[0][1] - a[1][1]]

    #data + occlusion
    def data_occlusion_value(self, asn, K, debug=False):
        p,q = asn
        lwindow = self.limg_padded[ p[0]:p[0]+2*self.windowsize+1,
                                    p[1]:p[1]+2*self.windowsize+1]
        rwindow = self.rimg_padded[ q[0]:q[0]+2*self.windowsize+1,
                                    q[1]:q[1]+2*self.windowsize+1]
        if debug:
            print('lwindow',lwindow)
            print('rwindow',rwindow)
        abs_difference = np.abs(lwindow-rwindow)
        abs_difference_trimmed = abs_difference * (abs_difference<=30) + 30 * (abs_difference>30)
        E_data = np.sum(abs_difference_trimmed ** self.d)
        return E_data - K

    def build_node_with_data_occlusion(self, p, alpha):
        '''
        creates appropriate nodes and adds them to varsA and vars0
        p: pixel in the left image
        alpha: alpha-expansion disparity value
        '''
        d = self.d_left[p]
        if d == alpha:
            # a in A_alpha and is active remains active
            asn = (p, (p[0], p[1]+d))
            Dp = self.data_occlusion_value(asn, self.K)
            self.energy.add_constant(Dp)
            self.varsA[p] = VAR_ALPHA
            self.vars0[p] = VAR_ALPHA
            #a in A_alpha and is inative changes depending on g
        else:
            if p[1]+alpha<self.limg.shape[1]:
                # p has inactive assignnent with disparity = alpha
                asn = (p, (p[0], p[1]+alpha))
                Dp = self.data_occlusion_value(asn, self.K)
                a = self.energy.add_node(0, Dp)
                self.varsA[p] = a
            else:
                # p+alpha is not in right image
                self.varsA[p] = VAR_ABSENT
            
            if d != OCCLUDED:
                # p has active assignemnt with disparity not= alpha
                asn = (p, (p[0], p[1]+d))
                Dp = self.data_occlusion_value(asn, self.K)
                o = self.energy.add_node(Dp, 0)
                self.vars0[p] = o
            else:
                # p has no active assignment
                self.vars0[p] = VAR_ABSENT

    def smoothness_value(self, asn1, asn2, lam):
        p1 = self.limg[asn1[0]]
        p2 = self.limg[asn2[0]]
        q1 = self.limg[asn1[1]]
        q2 = self.limg[asn2[1]]
        if(max(abs(p1-p2), abs(q1-q2))<8):
            return 3*lam
        else:
            return lam
        
    def build_smoothness(self, p1, p2, alpha):
        a1 = self.varsA[p1]
        a2 = self.varsA[p2]
        o1 = self.vars0[p1]
        o2 = self.vars0[p2]
        d1 = self.d_left[p1]
        d2 = self.d_left[p2]

        # by the definition of the smoothness term, we only care about p1 and p2 if they are adjacent
        # pixel. The assignments concerning p1 and p2 should have the same disparity, since the al-
        # -pha expansion move only changes inactive assignment with disp=alpha and active assignment
        # with disp!=alpha, we only concern ourselves with assignment disp=alpha and disp= active-ass-
        # ignment_of(p1, p2)

        #disp = alpha
        if(self.varsA[p1]!=VAR_ABSENT and self.varsA[p2]!=VAR_ABSENT):
            # both p1 and p2 has assignment with disp=alpha in rimg
            # at least one of p1 and p2 has inactive assignment with disp=alpha
            asn1 = (p1, (p1[0], p1[1]+alpha))
            asn2 = (p2, (p2[0], p2[1]+alpha))
            V = self.smoothness_value(asn1, asn2, self.lam)
            if(a1!=VAR_ALPHA and a2!=VAR_ALPHA):
                self.energy.add_term2(a1,a2,0,V,V,0)
            elif(a1!=VAR_ALPHA and a2==VAR_ALPHA):
                self.energy.add_term1(a1,V,0)
            elif(a1==VAR_ALPHA and a2!=VAR_ALPHA):
                self.energy.add_term1(a2,V,0)

        #disp= active-assignment_of(p1)=active-assignment_of(p2)
        if(d1==d2 and IS_VAR(o1) and IS_VAR(o2)):
            asn1 = (p1, (p1[0], p1[1]+d1))
            asn2 = (p2, (p2[0], p2[1]+d2))
            V = self.smoothness_value(asn1, asn2, self.lam)
            self.energy.add_term2(o1, o2, 0, V, V, 0)

        #disp= active-assignment_of(p1)!=active-assignment_of(p2)
        if(d1!=d2 and IS_VAR(o1) and p2[1]+d1<self.limg.shape[1]):
            asn1 = (p1, (p1[0], p1[1]+d1))
            asn2 = (p2, (p2[0], p2[1]+d1))
            V = self.smoothness_value(asn1, asn2, self.lam)
            self.energy.add_term1(o1, V, 0)
        
        #disp= active-assignment_of(p2)!=active-assignment_of(p1)
        if(d2!=d1 and IS_VAR(o2) and p1[1]+d2<self.limg.shape[1]):
            asn1 = (p1, (p1[0], p1[1]+d2))
            asn2 = (p2, (p2[0], p2[1]+d2))
            V = self.smoothness_value(asn1, asn2, self.lam)
            self.energy.add_term1(o2, V, 0)
            
        return


    def build_uniqueness(self, p, alpha):
        # Build edges in graph enforcing uniqueness at pixels p and p+d:
        # - prevent (p, p+d) and (p, p+a) from being both active.
        # - prevent (p, p+d) and (p+d-alpha, p+d) from being both active.
        o = self.vars0[p]

        if not IS_VAR(o):
            return
        
        # enforce unique image of p
        a = self.varsA[p]
        if a != VAR_ABSENT:
            self.energy.forbid01(o, a)

        # enforce unique antecedent of p+d
        d = self.d_left[p]
        # d cannot be OCCLUDED
        p2 = (p[0],p[1] + d - alpha)
        if 0 <= p[1] <= self.limg.shape[1]:
            a2 = self.varsA[p2]
            # a cannot be active
            self.energy.forbid01(o, a2)

    def neighbors(self, p):
        p_1, p_2 = p
        neighbors = [(p_1+1, p_2), (p_1, p_2+1), (p_1+1, p_2+1)]
        return neighbors
    
    def build_energy(self, alpha):
        self.energy = Energy()
        for i in range(self.limg.shape[0]):
            for j in range(self.limg.shape[1]):
                p =(i,j)
                self.build_node_with_data_occlusion(p, alpha)
                self.build_uniqueness(p, alpha)

        # all neighbors in bound
        for i in range(self.limg.shape[0]-1):
            for j in range(self.limg.shape[1]-1):
                p1 =(i,j)
                for p2 in self.neighbors(p1):
                    self.build_smoothness(p1, p2, alpha)

        # right neighbors out of bound
        for i in range(self.limg.shape[0]-1):
            # right-most column
            j = self.limg.shape[1] - 1
            p1 = (i, j)
            p2 = (i-1, j)
            self.build_smoothness(p1, p2, alpha)

        # down neighbors out of bound
        for j in range(self.limg.shape[1]-1):
            # bottom column
            i = self.limg.shape[0] - 1
            p1 = (i, j)
            p2 = (i, j + 1)
            self.build_smoothness(p1, p2, alpha)

    
    def minimize_energy(self):
        return self.energy.minimize()

    def recover_configuration(self, alpha):
        partition = self.energy.mincut.partition
        g_alpha = np.zeros(self.energy.vertex_num)
        g_alpha[partition[1]] = 1
        # print(partition)
        for i in range(self.limg.shape[0]):
            for j in range(self.limg.shape[1]):
                p = (i,j)
                alpha_node_id = self.varsA[p]
                d_node_id = self.vars0[p]
                if IS_VAR(d_node_id) and g_alpha[int(d_node_id)]==1:
                    self.d_left[p] = OCCLUDED
                if IS_VAR(alpha_node_id) and g_alpha[int(alpha_node_id)]==1:
                    self.d_left[p] = alpha
        self.d_right_from_d_left()
                
    def d_right_from_d_left(self):
        for i in range(self.limg.shape[0]):
            for j in range(self.limg.shape[1]):
                p = (i,j)
                dl = self.d_left[p]
                if(dl != OCCLUDED):
                    self.d_right[p[0], p[0]+dl] = -dl

    def expansion_move(self, alpha):
        # Compute the minimum alpha-expansion configuration
        # Return whether the move is different from previous E

        # build graph
        self.build_energy(alpha=alpha)

        oldE = self.preE
        # Max-flow, give the lowest-energy expansion move
        E = self.minimize_energy()
        # print("new E", E)
        if E < oldE:
            # lower energy, accept the expansion move
            self.recover_configuration(alpha)
            self.preE = E
            return True
        return False

    def tune_K(self):
        lower_one_fourth = int(self.maxdisparity/4)
        lower_one_fourth_sum = 0
        for i in range(self.limg.shape[0]):
            for j in range(self.limg.shape[1]):
                d_array = []
                for d in range(min(self.maxdisparity, self.limg.shape[1] - j)):
                    asn = ((i,j),(i,j+d))
                    d_array.append(self.data_occlusion_value(asn, 0))
                d_array.sort()
                lower_one_fourth_sum += np.sum(d_array[0:lower_one_fourth])
        self.K = lower_one_fourth_sum/(self.limg.shape[0]*self.limg.shape[1]*lower_one_fourth)
        self.lam = self.K/5
                

    def run(self):
        # Alpha-expansion move
        # assume min disparity = 0
        # randomize the order of alpha in each iter

        done = np.full((self.maxdisparity), 0)
        for iter in range(MAX_ITER):
            if np.sum(done) == done.shape[0]:
                break
            # randomize every iteration
            permutation = np.random.permutation(self.maxdisparity)
            for i in range(self.maxdisparity):
                alpha = permutation[i]
                if done[alpha]:
                    continue
                # calculate min energy for this alpha
                # print("current E", self.preE)
                if self.expansion_move(alpha):
                    done = np.full((self.maxdisparity), 0)
                done[alpha] = 1

    def add_smoothness(self, p1, p2):
            smoothness_term = 0
            d1 = self.d_left[p1]
            d2 = self.d_left[p2]
            if d1 == d2:
                return 0
            if d1 != OCCLUDED and \
                p2[1] + d1 < self.limg.shape[1]:
                smoothness_term += self.smoothness_value((p1, (p1[0], p1[1] + d1)), (p2, (p2[0], p2[1] + d1)), self.lam)
            
            if d2 != OCCLUDED and \
                p1[1] + d2 < self.limg.shape[1]:
                smoothness_term += self.smoothness_value((p1, (p1[0], p1[1] + d2)), (p2, (p2[0], p2[1] + d2)), self.lam)
            
            return smoothness_term

    def ComputeEnergy(self):
        
        # Compute current minimized energy
        # only use for debugging
        E = 0
        dovcount = 0
        activecount = 0
        for i in range(self.limg.shape[0]):
            for j in range(self.limg.shape[1]):
                p1 = (i, j)
                d1 = self.d_left[p1]
                if d1 != OCCLUDED:
                    dov = self.data_occlusion_value((p1, (i, j + d1)), self.K)
                    if(dov<0):
                        dovcount+=1
                    E += dov
                    activecount += 1
        print("dovcount", dovcount)
        print("activecount", activecount)
        print("E from data occlusion", E)
        E0 = E
        # all neighbors in bound
        for i in range(self.limg.shape[0]-1):
            for j in range(self.limg.shape[1]-1):
                p1 =(i,j)
                for p2 in self.neighbors(p1):
                    E += self.add_smoothness(p1, p2)


        # right neighbors out of bound
        for i in range(self.limg.shape[0]-1):
            # right-most column
            j = self.limg.shape[1] - 1
            p1 = (i, j)
            p2 = (i-1, j)
            E += self.add_smoothness(p1, p2)

        # down neighbors out of bound
        for j in range(self.limg.shape[1]-1):
            # bottom column
            i = self.limg.shape[0] - 1
            p1 = (i, j)
            p2 = (i, j + 1)
            E += self.add_smoothness(p1, p2)
        
        
        print("E from smoothness", E-E0)
        return E
