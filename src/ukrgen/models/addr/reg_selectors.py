import logging

from ..lsc.offset import lsc_offset

class addr_reg_selector:
    """
    Class for selecting address registers
    """
    def reset(self,
              current_index : int,
              current_offset : lsc_offset,
              target_offset : lsc_offset,
              offset_range : tuple[lsc_offset,lsc_offset]):
        """
        Called at the beginning of the search, the parameters passed
        are for the first register in the list
        """
        pass

    def __call__(self,
              current_index : int,
              current_offset : lsc_offset,
              target_offset : lsc_offset,
              offset_range : tuple[lsc_offset,lsc_offset]) -> bool:
        """
        Called for each address register, needs to return True if the register
        is a good candidate and false if it isn't
        """
        pass

# Example expected behaviours:
# a1 : 0
# a2 : 2
# range: [0,1]
# t  : 0,1,2,3,4,5,6,7
# a1+0,a1+1,a2,a2+1, (add a1+4,a1+0), a1+1, (add a2+4,a2+0), a2+1
#
# a1 : 0
# a2 : 1
# range: [0,2]
# t  : 0,1,2,3
# a1+0,a2+0,a1+2,a2+2, (add a1+4,a1+0), (add a2+4, a2+0), a1+2, a2+2


# reset: idx 0, off 0, t 0, r [0,2]
#   a1 : idx 0, off 0, t 0, r [0,2], decide true
#   a2 : idx 1, off 1, t 0, r [0,2], decide false
# reset: idx 0, off 0, t 1, r [0,2]
#   a1 : idx 0, off 0, t 1, r [0,2], decide false or true (will be overwritten)
#   a2 : idx 1, off 1, t 1, r [0,2], decide true
# reset: idx 0, off 0, t 2, r [0,2]
#   a1 : idx 0, off 0, t 2, r [0,2], decide true
#   a2 : idx 1, off 1, t 2, r [0,2], decide false
# reset: idx 0, off 0, t 3, r [0,2]
#   a1 : idx 0, off 0, t 3, r [0,2], decide false or true
#   a2 : idx 1, off 1, t 3, r [0,2], decide true
# reset: idx 0, off 0, t 4, r [0,2]
#   a1 : idx 0, off 0, t 4, r [0,2], decide true
#   a2 : idx 1, off 1, t 4, r [0,2], decide false
# (this decision will cause a1 to be incremented by 4 elsewhere)
# reset: idx 0, off 4, t 5, r [0,2]
#   a1 : idx 0, off 4, t 5, r [0,2], decide false or true (will be overwritten)
#   a2 : idx 1, off 1, t 5, r [0,2], decide true
# (this decision will cause a2 to be incremented by 4 elsewhere)
# reset: idx 0, off 4, t 6, r [0,2]
#   a1 : idx 0, off 4, t 6, r [0,2], decide true
#   a2 : idx 1, off 5, t 6, r [0,2], decide false
# reset: idx 0, off 4, t 7, r [0,2]
#   a1 : idx 0, off 4, t 7, r [0,2], decide false or true (will be overwritten)
#   a2 : idx 1, off 5, t 7, r [0,2], decide true

class rotating_selector(addr_reg_selector):
    """
    Selects the candidate so that address registers are rotated through
    """
    def __init__(self):
        self.perfect_index = None
        self.subperfect_index = None
        self.oor_index = None
        self.best_index = None

        self.smallest_offset = None
        self.smallest_offset_index = None

        self.unrotated = set()
        self.rotating = False


    def reset(self,
              current_index : int,
              current_offset : lsc_offset,
              target_offset : lsc_offset,
              offset_range : tuple[lsc_offset,lsc_offset]):
        self.perfect_index = None
        self.subperfect_index = None

        # best index was oor, start rotating
        if self.oor_index is not None and\
                self.oor_index == self.best_index:
            self.rotating = True
        # if we're not rotating add the previously selected index to unrotated set
        if not self.rotating and self.best_index is not None:
            self.unrotated.add(self.best_index)

        # reset best index
        self.best_index = None

        # if we're rotating remove the smallest index (will be the one chosen)
        if self.rotating:
            if self.smallest_offset_index in self.unrotated:
                self.unrotated.remove(self.smallest_offset_index)

        # if no indices left in unrotated set, we're finished rotating
        if not self.unrotated:
            self.oor_index = None
            self.rotating = False

        self.smallest_offset = current_offset
        self.smallest_offset_index = current_index

        if current_offset == target_offset:
            self.perfect_index = current_index
        if (target_offset-current_offset) < offset_range[1]:
            self.subperfect_index = current_index


    def __call__(self,
                 current_index : int,
                 current_offset : lsc_offset,
                 target_offset : lsc_offset,
                 offset_range : tuple[lsc_offset,lsc_offset]) -> bool:


        in_range = current_offset.in_range_of(target_offset, offset_range)

        is_smallest = current_offset < self.smallest_offset\
                   or current_offset == self.smallest_offset

        if is_smallest:
            self.smallest_offset = current_offset
            self.smallest_offset_index = current_index


        # exact match
        if self.perfect_index is not None:
            if current_index == self.perfect_index:
                self.best_index = current_index
                return True
            return False
        if current_offset == target_offset:
            self.perfect_index = current_index
            self.best_index = current_index
            return True
        
        if self.rotating:
            if is_smallest:
                return True
            return False


        # not exactly matching but in range
        if self.subperfect_index is not None:
            if current_index == self.subperfect_index:
                self.smallest_offset = current_offset
                self.best_index = current_index
                return True
            if is_smallest and in_range:
                self.smallest_offset = current_offset
                self.best_index = current_index
                self.subperfect_index = current_index
                return True
            return False
        if in_range:
            self.smallest_offset = current_offset
            self.subperfect_index = current_index
            self.best_index = current_index
            return True

        # not in range, return True for the lowest offset
        if is_smallest:
            self.oor_index = current_index
            self.best_index = current_index
            return True

        return False


class mindist_selector(addr_reg_selector):
    def __init__(self):

        self.dist_min = None

    def reset(self,
              current_index : int,
              current_offset : lsc_offset,
              target_offset : lsc_offset,
              offset_range : tuple[lsc_offset,lsc_offset]):
        self.dist_min = abs(target_offset - current_offset)

    def __call__(self,
                 current_index : int,
                 current_offset : lsc_offset,
                 target_offset : lsc_offset,
                 offset_range : tuple[lsc_offset,lsc_offset]) -> bool:

        dist = abs(target_offset-current_offset)
        if dist < self.dist_min:
            self.dist_min = dist
            return True
        return False

class phasing_selector(addr_reg_selector):
    """
    Selects candidates in phases, i.e. first one address register
    is used for all accesses, then the next when starting from
    the first offset again
    """
    def __init__(self):
        self.last_chosen = None
        self.current_candidate = None
        self.phases_done = set()
        self.phases_in_progress = set()

        self.debug = logging.getLogger("ADDR").debug


    def reset(self,
              current_index : int,
              current_offset : lsc_offset,
              target_offset : lsc_offset,
              offset_range : tuple[lsc_offset,lsc_offset]):
        

        if self.current_candidate is not None:
            self.phases_in_progress.add(self.current_candidate)

            if self.last_chosen is None:
                pass
            elif self.current_candidate != self.last_chosen:
                self.debug(f"********************************")
                self.debug(f"Phase {self.last_chosen} is done")
                self.debug(f"********************************")
                self.phases_done.add(self.last_chosen)
                self.phases_in_progress.remove(self.last_chosen)

        self.last_chosen = self.current_candidate
        self.current_candidate = current_index
        

    def __call__(self,
                 current_index : int,
                 current_offset : lsc_offset,
                 target_offset : lsc_offset,
                 offset_range : tuple[lsc_offset,lsc_offset]) -> bool:
        
        choose = False


        # prefer exact match
        if current_offset == target_offset:
            choose = True
        # prefer last chosen
        elif current_index == self.last_chosen:
            choose = True

        # Don't choose completed phases
        if current_index in self.phases_done:
            choose = False
        

        if choose:
            self.current_candidate = current_index

        return choose
