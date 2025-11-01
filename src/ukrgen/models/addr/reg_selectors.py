from ..lsc.offset import lsc_offset

class interleaving_selector:
    """
    Selects the candidate so that address register use is interleaved
    
    By recommending the candidate with the "smallest" offset
    the interleaving is implicit
    """
    def __init__(self):
        self.offset_min = None
        self.this_best = None

    def reset(self,
              current_offset : lsc_offset,
              target_offset : lsc_offset,
              offset_range : tuple[lsc_offset,lsc_offset]):
        self.offset_min = current_offset
        self.dist_min = None
        self.rollover = True
        self.locked_in = False
        self.last_best = self.this_best

    def __call__(self,
                 current_offset : lsc_offset,
                 target_offset : lsc_offset,
                 offset_range : tuple[lsc_offset,lsc_offset]) -> bool:

        if self.locked_in:
            print(f"Skipping because of perfect choice")
            return False

        if current_offset == target_offset:
            self.this_best = current_offset
            self.locked_in = True
            print(f"Perfect choice")
            return True

        if current_offset == self.last_best:
            print(f"Skipping because it's the last best")
            return False

        if current_offset > self.last_best:
            print("larger than last best offset encountered, cancelling rollover")
            self.rollover = False
        
        dist = target_offset - current_offset

        if self.rollover:
            if (self.offset_min is None) or not (current_offset > self.offset_min):
                self.offset_min = current_offset
                self.this_best = current_offset
                print(f"Chosen smallest after rollover")
                return True
            else:
                print(f"off {current_offset} >= {self.offset_min}; skipping")
        else:
            if (self.dist_min is None) or (dist < self.dist_min):
                self.dist_min = dist
                self.this_best = current_offset
                print(f"Chosen smallest distance before rollover")
                return True
            else:
                print(f"dist {dist} >= {self.dist_min}; skipping")

        print(f"Skipped as no condition applied")
        return False

class mindist_selector:
    def __init__(self):

        self.dist_min = None

    def reset(self,
              current_offset : lsc_offset,
              target_offset : lsc_offset,
              offset_range : tuple[lsc_offset,lsc_offset]):
        self.dist_min = abs(target_offset - current_offset)

    def __call__(self,
                 current_offset : lsc_offset,
                 target_offset : lsc_offset,
                 offset_range : tuple[lsc_offset,lsc_offset]) -> bool:

        dist = abs(target_offset-current_offset)
        if dist < self.dist_min:
            self.dist_min = dist
            return True
        return False

class phasing_selector:
    """
    Selects candidates in phases, i.e. first one address register
    is used for all accesses, then the next when starting from
    the first offset again
    """
    def __init__(self):
        self.last_target = None
        self.this_target = None


    def reset(self,
              current_offset : lsc_offset,
              target_offset : lsc_offset,
              offset_range : tuple[lsc_offset,lsc_offset]):
        
        self.last_target = self.this_target
        

    def __call__(self,
                 current_offset : lsc_offset,
                 target_offset : lsc_offset,
                 offset_range : tuple[lsc_offset,lsc_offset]) -> bool:


        if current_offset == target_offset:
            self.this_target = target_offset
            return True

        if current_offset == self.last_target:
            self.this_target = target_offset
            return True


        return False
