from common_package import *

def isvector(argument):

    if isinstance(argument,np.ndarray):
        return True

def isscalar(argument):

    if not isinstance(argument,np.ndarray):
        return False