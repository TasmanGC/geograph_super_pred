import  pandas as pd
from    tqdm import trange
from    sklearn.preprocessing import QuantileTransformer, minmax_scale
import  numpy as np
import sklearn

# TODO lets confirm the need for a normize function as here and either enforce the same usage throughout or remove this
def normalize(values, bounds={'lower':0,'upper':1}):
    return [bounds['lower'] + (x - min(values)) * (bounds['upper'] - bounds['lower']) / (max(values) - min(values)) for x in values]

class CoreNode:
    def __init__(self, loc, params):
        """ This class is a universal node structure.
        
        Contains common parameters accross both input datasets.

        :param iD: global id, unique to the node
        :param loc: 3D location of the given ndoe
        :param params: dictionary containing all the values of note
        :type iD: int
        :type loc: list of nums with length of 3
        
        .. seealso:: :class:`Trace` :class:`AEMStation`
        """
        self.nloc       = loc
        self.params     = params
        
class Station:
    def __init__(self,iD,loc,ntype,params={}):
        self.iD     = iD
        self.loc    = loc  
        self.params = params
        self.type   = ntype
        self.pred   = {}

    def conv2core(self):
        """ This method converts the AEMStation into core node structures.

        This method performs the simply divides AEMStation into nodes.

        :returns CoreNodeList:
        :rtype: list
        
        .. seealso:: :class:`CoreNode`
        """
        x = self.loc[0]
        y = self.loc[1]

        CoreNodeList = []
        for i in range(len(self.params['depth'])): # TODO update
            # generate a node_iD
            # extract parameters
            z = self.params["depth"][i]
            r = self.params["rdata"][i]
            l = self.params["label"][i]

            params = {
                        "nodet" : self.type,
                        "rdata" : r,
                        "depth" : z,
                        "label" : l,
                    }

            CoreNodeList.append(CoreNode([x,y,z],params))

        return(CoreNodeList)