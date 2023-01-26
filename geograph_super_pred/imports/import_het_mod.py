import numpy as np
from tqdm import trange
   

def station_links(atype, btype, num_b = 1):
    '''
    atype stations are connected to num_b btype stations
    '''
    counter = trange(len(atype), desc='Linking Data',leave=True)
    
    # we modify our traces in place
    for i in counter:
        
        counter.set_description('Linking Data - (Graphs %i)' % i)
        t_station = atype[i]

        s_loc = np.array(t_station.loc[:2]) # seismic surface location

        # ----------------- Identify the two cloest b stations to our selected trace ------------------- # 
        
        cl1_loc = np.array([x.loc[:2] for x in btype])                                    # 0.2 - isolate their locations
        cl1_dis = np.linalg.norm(s_loc - cl1_loc,   axis=1)                                 # 0.3 - calculate the aem station with the closest minimum distance
        cl1_idx = list(np.argpartition(cl1_dis, num_b))[:num_b]                             # 0.4 - get the line dependant index of the aem station
        cl1_sel = np.array([station for i, station in enumerate(btype) if i in cl1_idx])  # 0.5 - select the n_aem cloest stations 

        t_station.params['b_cons'] = cl1_sel
        
    return(atype)