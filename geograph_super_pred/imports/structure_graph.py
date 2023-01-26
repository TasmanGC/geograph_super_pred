import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from    torch.utils.data import Dataset
import  dgl
from    dgl.data import DGLDataset
from    dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader # these are used in another script but should be imported here
import  numpy as np
from    random import sample

def het_stat_graph(a_mod, nodes_dict={'A':None,'B':None}, window={'A':2,'B':2}, stride={'A':1,'B':1}):
    
    # we generate the core nodes to put in the graph
    graph_nodes = {}
    graph_nodes['A'] = a_mod.conv2core()
    B_nodes = []
    for stat in a_mod.params['b_cons']:
        B_nodes.extend(stat.conv2core())
    graph_nodes['B'] = B_nodes
    num_B_stat = len(list(a_mod.params['b_cons']))

    # if we want we can subsample these nodes by a given factor
    for ntype, factor in nodes_dict.items():
        if isinstance(factor, int):
            graph_nodes[ntype] = graph_nodes[ntype][0::factor]
        else:
            pass
    
    # now lets generate some local graph ids
    # these are used from here on as the verticies ids
    for node_list in graph_nodes.values():
        for i, node in enumerate(node_list):
            node.params['local_id'] = i

    # edge dictionary
    edge_dict = {}
    # same
    edge_dict['A_same'] = [[],[]]
    edge_dict['B_same'] = [[],[]]

    # adjacent
    for ntype, n_list in graph_nodes.items():
        vert_u = []
        vert_v = []
        if ntype=='A':
            for i, _ in enumerate(n_list):
                if i < len(n_list)-1:
                    vert_u.append(n_list[i].params['local_id'])
                    vert_v.append(n_list[i+1].params['local_id'])
            edge_dict[ntype+'_same'][0].extend(vert_u)
            edge_dict[ntype+'_same'][1].extend(vert_v)

        if ntype=='B':
            npstat = int(len(n_list)/num_B_stat) # nodes per B_station
            station_nodes = [list(x) for x in np.reshape(list(n_list),[-1,npstat])]
            for s_list in station_nodes:
                for i, _ in enumerate(s_list):
                    if i < len(s_list)-1:
                        vert_u.append(s_list[i].params['local_id'])
                        vert_v.append(s_list[i+1].params['local_id'])    
            edge_dict[ntype+'_same'][0].extend(vert_u)
            edge_dict[ntype+'_same'][1].extend(vert_v)    

    #window
    for ntype, n_list in graph_nodes.items():
        vert_u = []
        vert_v = []

        if ntype=='A':
            for _, n in enumerate(n_list):
                vert_u = [[n.params['local_id']+x,n.params['local_id']-x] for x in range(0,window[ntype]*stride[ntype],stride[ntype])] # get the ids in a window
                vert_u = list(set([x for xs in vert_u for x in xs]))           # flatten the cons
                stat_u = [n.params['local_id'] for n in n_list]     # list of acceptable ids
                vert_u = [u for u in vert_u if u in stat_u]    # keep only the acceptable verticies
                vert_v = [n.params['local_id']]*len(vert_u)         # make an equal ammount of source_ids
                
                edge_dict[ntype+'_same'][0].extend(vert_u)
                edge_dict[ntype+'_same'][1].extend(vert_v)  

        if ntype=='B':
            npstat = int(len(n_list)/num_B_stat) # nodes per aem_station
            station_nodes = [list(x) for x in np.reshape(list(n_list),[-1,npstat])]
            for n_list in station_nodes:
                for n in n_list:
                    stat_u = [n.params['local_id'] for n in n_list]      # list of acceptable ids
                    vert_u = [[n.params['local_id']+x,n.params['local_id']-x] for x in range(0,window[ntype]*stride[ntype],stride[ntype])] # get the ids in a window
                    vert_u = [x for xs in vert_u for x in xs]           # flatten the cons
                    vert_u = [u for u in vert_u if u in stat_u]    # keep only the acceptable verticies
                    vert_v = [n.params['local_id']]*len(vert_u)         # make an equal ammount of source_ids
                    edge_dict[ntype+'_same'][0].extend(vert_u)
                    edge_dict[ntype+'_same'][1].extend(vert_v)  

    # diff
    edge_dict['node_dif'] = [[],[]]

    # lets split the flat B_list into n stations
    a_step = graph_nodes['B']
    b_step = np.reshape(a_step,(-1,int(len(graph_nodes['B'])/num_B_stat)))
    c_step = list(b_step)
    split_B = [list(a) for a in c_step]


    for A_c in graph_nodes['A']: # for every seismic node

        for B_stat in split_B:
            # calculate the distance between our current seismic node and the current aem stations core nodes
            stat_nodes          = np.array([np.array(n.nloc) for n in B_stat])
            stat_nodes_dist     = np.linalg.norm(np.array(A_c.nloc)  - stat_nodes, axis=1)
            # index of closest core node of our current aem station
            q                   = np.argmin(stat_nodes_dist)

            # append the id of the seismic node we're looking at and the aem node that's closest
            edge_dict['node_dif'][0].append(A_c.params['local_id'])       # seismic id
            edge_dict['node_dif'][1].append(B_stat[q].params['local_id']) # aem id


    all_A_ids = list(np.unique(edge_dict['A_same'][0] + edge_dict['A_same'][1] + edge_dict['node_dif'][0]))
    all_B_ids = list(np.unique(edge_dict['B_same'][0] + edge_dict['B_same'][1] + edge_dict['node_dif'][1]))
   

    # feat definitions
    A_data = [n.params['rdata']          for n in graph_nodes['A'] if n.params['local_id'] in all_A_ids]
    A_type = [0                          for n in graph_nodes['A'] if n.params['local_id'] in all_A_ids]
    A_nloc = [n.nloc                     for n in graph_nodes['A'] if n.params['local_id'] in all_A_ids]
    A_labl = [n.params['label']          for n in graph_nodes['A'] if n.params['local_id'] in all_A_ids]

    B_data = [n.params['rdata']          for n in graph_nodes['B'] if n.params['local_id'] in all_B_ids]
    B_type = [1                          for n in graph_nodes['B'] if n.params['local_id'] in all_B_ids]
    B_nloc = [n.nloc                     for n in graph_nodes['B'] if n.params['local_id'] in all_B_ids]
    B_labl = [n.params['label']          for n in graph_nodes['B'] if n.params['local_id'] in all_B_ids]

    # generate and populate our bipartite graph
    graph = dgl.heterograph({   ('A', 'same_a', 'A'): (edge_dict['A_same'][0], edge_dict['A_same'][1]),
                                ('B', 'same_b', 'B'): (edge_dict['B_same'][0], edge_dict['B_same'][1]),
                                ('A', 'diff_a', 'B'): (edge_dict['node_dif'][0], edge_dict['node_dif'][1]),
                                ('B', 'diff_b', 'A'): (edge_dict['node_dif'][1], edge_dict['node_dif'][0])} )

    graph.nodes['B'].data['data']      = torch.tensor(np.array(B_data), dtype=torch.float32).unsqueeze(1)
    graph.nodes['B'].data['labl']      = torch.tensor(np.array(B_labl), dtype=torch.float32)
    graph.nodes['B'].data['nloc']      = torch.tensor(np.array(B_nloc), dtype=torch.float32)
    graph.nodes['B'].data['ntyp']      = torch.tensor(np.array(B_type), dtype=torch.float32)
    graph.nodes['A'].data['data']      = torch.tensor(np.array(A_data), dtype=torch.float32).unsqueeze(1)
    graph.nodes['A'].data['labl']      = torch.tensor(np.array(A_labl), dtype=torch.float32)
    graph.nodes['A'].data['nloc']      = torch.tensor(np.array(A_nloc), dtype=torch.float32)
    graph.nodes['A'].data['ntyp']      = torch.tensor(np.array(A_type), dtype=torch.float32)

    return(graph)


def stat_graph(stat,edge_methods = ['window','time'],window=1,stride=1,subf= 1):

    graph_nodes = stat.conv2core() 
    graph_nodes = graph_nodes[0::subf]

    stat_u = []
    stat_v = []

    if 'window' in edge_methods:
        for i in range(len(graph_nodes)):
            b = [v for v in [item for sublist in [[i+x,i-x] for x in range(0,window*stride,stride)] for item in sublist] if v in list(range(len(graph_nodes)))]
            stat_u.extend(b)
            stat_v.extend([i]*len(b))

    if 'time' in edge_methods:
            for i in range(len(graph_nodes)):
                if i < len(range(len(graph_nodes)))-1:
                    stat_u.append(i)
                    stat_v.append(i+1)

    # dictionary for mapping prospect
    p_di = {'195Z':0, 'CNT':1,'275Z':2}

    # feat definitions
    stat_data = stat.params['rdata'][0::subf]
    stat_type = [0 if stat.type=='A' else 1 for n in graph_nodes]
    stat_nloc = [n.nloc                     for n in graph_nodes]
    stat_labl = [n.params['label']          for n in graph_nodes]

    # thus we need increase the aemids by the number of seismic nodes, and reindex them to ensure we're not creating isolated duplicate nodes

    feat = stat_data
    ntyp = stat_type
    nloc = stat_nloc
    labl = stat_labl


    # generate and populate our bipartite graph
    graph = dgl.graph((stat_u, stat_v))

    graph.ndata['data'] = torch.tensor(np.array(feat), dtype=torch.float32).unsqueeze(1)
    graph.ndata['labl'] = torch.tensor(np.array(labl), dtype=torch.float32).unsqueeze(1)
    graph.ndata['nloc'] = torch.tensor(np.array(nloc), dtype=torch.float32).unsqueeze(1)
    graph.ndata['ntyp'] = torch.tensor(np.array(ntyp), dtype=torch.float32).unsqueeze(1)    
    return(graph)