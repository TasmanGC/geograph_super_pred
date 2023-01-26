
from .import_classes import *
from .import_het_mod import *
from .structure_graph import *
from .import_synthdata import *

from    tqdm import tqdm
import  torch
import  dgl
from    dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

# TODO lets confirm the need for a normize function as here and either enforce the same usage throughout or remove this
def normalize(values, bounds={'lower':0,'upper':1}):
    return [bounds['lower'] + (x - min(values)) * (bounds['upper'] - bounds['lower']) / (max(values) - min(values)) for x in values]

def gen_data_loader(train_graph, type, device, B_SIZE):


    if type!='het':

        node_ids    = torch.arange(0,train_graph.num_nodes(),dtype=torch.int64).to(device) # train on all nodes of graph 1

    if type=='het':

        node_ids = {}
        node_ids['B'] = torch.arange(0,train_graph.num_nodes('B'),dtype=torch.int64).to(device)
        node_ids['A'] = torch.arange(0,train_graph.num_nodes('A'),dtype=torch.int64).to(device)

    # samples graphs by extraacting nodes in a 2 hop neighbor hood for a given target node
    sampler         = MultiLayerFullNeighborSampler(2)
    data_loader     = DataLoader(train_graph, node_ids, sampler, batch_size=B_SIZE, shuffle=True, drop_last=False,  device=device)

    return(data_loader)

def batched_graph(basement_df, mode='het'):

    # synthetic stations of two different basement expressions
    a_stations = generate_synth_stations(basement_df, 100, vert_res=20, feat_type='A') # note here we can experiment with the structure of the data
    b_stations = generate_synth_stations(basement_df, 100, vert_res=50, feat_type='B')

    # link the two station types
    atype_links = station_links(a_stations,b_stations, num_b = 1)

    # construct a series of dgl graphs
    graphs = []

    if mode=='het': 
        for stat in atype_links: # works with the updated linked data
            demo_het    = het_stat_graph(stat)
            graphs.append(demo_het)

    if mode=='A':
        for stat in a_stations: # works with just the a stations
            demo_het    = stat_graph(stat)
            graphs.append(demo_het)

    if mode=='B':
        for stat in b_stations: 
            demo_het    = stat_graph(stat)
            graphs.append(demo_het)
            
    # batch these graphs
    return(dgl.batch(graphs))