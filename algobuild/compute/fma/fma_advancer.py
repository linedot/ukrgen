import abc


from .fma import fma_info

class fma_advancer:
    
    @abc.abstractmethod
    def advance(self):
        raise NotImplementedError("Attempted to call abstract fma_advancer.advancer() method")


class fma_vector_mxn_advancer(fma_advancer):
    def __init__(self, 
                 a_vectors : list[int],
                 b_vectors : list[int],
                 c_vectors : list[int],
                 m : int, n : int):
        self.a_vectors = a_vectors
        self.b_vectors = b_vectors
        self.c_vectors = c_vectors
        self.a_idx = 0
        self.b_idx = 0
        self.c_idx = 0
        self.m = m
        self.n = n

        self.a_idx_offset = 0
        self.b_idx_offset = 0
        
        assert(len(self.c_vectors) == m*n, "Number of vectors in c-tile")


    @property
    def a_vector(self):
        idx = (self.a_idx + self.a_idx_offset) % len(self.a_vectors)
        return self.a_vectors[idx]

    @property
    def b_vector(self):
        idx = (self.b_idx + self.b_idx_offset) % len(self.b_vectors)
        return self.b_vectors[idx]

    @property
    def c_vector(self):
        return self.c_vectors[self.c_idx]

class fma_vector_mxn_afirst_advancer(fma_vector_mxn_advancer):
    
    def advance(self):
        self.c_idx = ((self.c_idx + 1) % (self.m*self.n))
        if 0 == self.c_idx:
            self.a_idx_offset += self.m
            self.a_idx_offset = self.a_idx_offset % len(self.a_vectors)
            self.b_idx_offset += self.n
            self.b_idx_offset = self.b_idx_offset % len(self.b_vectors)
        self.a_idx = ((self.a_idx + 1) % self.m)
        if 0 == self.a_idx:
            self.b_idx = ((self.b_idx + 1) % self.n) 

class fma_vector_mxn_bfirst_advancer(fma_vector_mxn_advancer):
    
    def advance(self):
        self.c_idx = ((self.c_idx + 1) % (self.m*self.n))
        if 0 == self.c_idx:
            self.a_idx_offset += self.m
            self.a_idx_offset = self.a_idx_offset % len(self.a_vectors)
            self.b_idx_offset += self.n
            self.b_idx_offset = self.b_idx_offset % len(self.b_vectors)
        self.b_idx = ((self.b_idx + 1) % self.n)
        if 0 == self.b_idx:
            self.a_idx = ((self.a_idx + 1) % self.m) 
