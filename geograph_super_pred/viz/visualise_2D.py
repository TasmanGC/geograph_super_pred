from   ..ml.ml_models       import HomoModel, HeteroModel
from   ..ml                 import CoreNode

import torch
import dgl
import rasterio
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
import verde                as vd

from   math                 import cos, sin
from   tqdm                 import tqdm
from   rasterio.transform   import Affine
from   dgl.dataloading      import MultiLayerFullNeighborSampler, DataLoader # these are used in another script but should be imported here

def df_to_geotif(file_name):

    d_frame = pd.read_csv(file_name)

    buffer = 10
    x       = d_frame.x 
    y       = d_frame.y

    # this we need to do twice
    for key in ['z_prd']:

        spline = vd.Spline(damping=0.0008)
        spline.fit((d_frame['x'],d_frame['y']), d_frame[key])

        grid_coords     = vd.grid_coordinates(region=(d_frame['x'].min()-buffer, d_frame['x'].max()+buffer,d_frame['y'].min()-buffer, d_frame['y'].max()+buffer), spacing=10)
        gridded_scalars = spline.predict(grid_coords)

    grid_coords[0], grid_coords[1]

    x_0 = grid_coords[0][0][0]
    x_N = grid_coords[0][0][-1]
    y_0 = grid_coords[1][0][0]
    y_N = grid_coords[1][-1][0]

    x_l, y_l = grid_coords[1].shape

    y_res = (y_N - y_0) / x_l
    x_res = (x_N - x_0) / y_l

    transform = Affine.translation(x_0 - x_res, y_0 - y_res) * Affine.scale(x_res, y_res)

    # create a geo-tif
    new_dataset = rasterio.open(
                                'new.tif',
                                'w',
                                driver='GTiff',
                                height=gridded_scalars.shape[0],
                                width=gridded_scalars.shape[1],
                                count=1,
                                dtype=gridded_scalars.dtype,
                                crs='EPSG:28350',
                                transform=transform,
                            )

    new_dataset.write(gridded_scalars, 1)

    new_dataset.close()

def reproject(x,y,angle):
    angle = np.deg2rad(angle)
    n_x = []
    n_y = []
    for oldX, oldY in zip(x,y):
        newX = oldX * cos(angle) - oldY * sin(angle)
        newY = oldX * sin(angle) + oldY * cos(angle)
        n_x.append(newX)
        n_y.append(newY)
    return(n_x,n_y)

def df_grid(d_frame,file_name):
    """ Performs simple spline based gridding of irregular point sampled data. 
    """
    fig, ax = plt.subplots(figsize=(8,8))

    buffer = 10
    x       = d_frame.x
    y       = d_frame.y

    for key in ['z_pred']:

        spline = vd.Spline(damping=0.0000008)
        spline.fit((d_frame['x'],d_frame['y']), d_frame[key])

        grid_coords     = vd.grid_coordinates(region=(d_frame['x'].min()-buffer, d_frame['x'].max()+buffer,d_frame['y'].min()-buffer, d_frame['y'].max()+buffer), spacing=50)
        gridded_scalars = spline.predict(grid_coords)

    pc0 = ax.pcolormesh(grid_coords[0], grid_coords[1], gridded_scalars)#cmap='cividis',
    fig.colorbar(pc0)
    ax.scatter(d_frame['x'], d_frame['y'],s=5,c='k',alpha=0.1) # plot the posting data
    #ax.set_ylim()
    
    plt.suptitle('Basment Surface')
    plt.tight_layout()
    plt.savefig(file_name)

def gen_test_dl(graph_list, batch_size=100, device=torch.device('cpu')):
    test__graph = dgl.batch(graph_list).to(device)

    if len(test__graph.ntypes)==1:
        test__ids = torch.arange(0,test__graph.num_nodes(),dtype=torch.int64).to(device)

    if len(test__graph.ntypes)>1:
        # test on all
        test__ids = {}
        test__ids['B'] = torch.arange(0,test__graph.num_nodes('B'),dtype=torch.int64).to(device)
        test__ids['A'] = torch.arange(0,test__graph.num_nodes('A'),dtype=torch.int64).to(device)

    sampler = MultiLayerFullNeighborSampler(2)
    test_loader     = DataLoader(test__graph, test__ids, sampler, batch_size = batch_size, shuffle=False, drop_last=False, device=device)
    return(test_loader)

# load models
def load_model(mod_file,out_size=1):

    if 'aem_model' in mod_file:
        model = HomoModel(3,130,out_size)

    if 'sei_model' in mod_file:
        model = HomoModel(1,130,out_size)
        
    if 'het_model' in mod_file:
        rel_dict = {'same_a' : 1, 'same_b' : 1, 'diff_a' : 1, 'diff_b' : 1}
        model = HeteroModel(rel_dict,130,out_size)
    
    model.load_state_dict(torch.load(mod_file))

    return(model)

def core_node_to_df(core_node_list):
    nde_npmy = np.array(core_node_list)                                 # list of objects
    loc_list = np.array([x.nloc[:2] for x in core_node_list])           # list of locations
    stations = np.unique([x.nloc[:2] for x in core_node_list],axis=0)   # list of unique stations

    x_loc = []
    y_loc = []
    z_tru = []
    z_prd = []

    for stat in tqdm(stations):
        x_loc.append(stat[0])
        y_loc.append(stat[1])
        z_tru_i = 0
        z_prd_i = 0

        loc_i = np.unique(np.where(loc_list==stat)[0])
        nodes = nde_npmy[loc_i]
        
        if nodes[0].params['type']=='B':
            pred = [x.params['pred'] for x in nodes] # don't repeat calculations for both connection dir
            labl = [x.params['labl'] for x in nodes] # don't repeat calculations for both connection dir
            try:
                #p_pick = pred.index(1)
                p_pick = len(pred) - 1 - pred[::-1].index(1)
            except:
                p_pick=0
            try:
                #l_pick = labl.index(1)
                l_pick = len(labl) - 1 - labl[::-1].index(1)
            except:
                l_pick=0
    
            pred_a = nodes[p_pick].nloc[2]
            true_a = nodes[l_pick].nloc[2]
            z_tru_i=true_a
            z_prd_i=pred_a

        if nodes[0].params['type']=='A':
            pred = [x.params['pred'] for x in nodes] # don't repeat calculations for both connection dir
            labl = [x.params['labl'] for x in nodes] # don't repeat calculations for both connection dir
            tar = [1]*3
            p_chck = [(i, i+len(tar)) for i in range(len(pred)-len(tar)+1) if pred[i:i+len(tar)] == tar]
            p_pick = 0 if len(p_chck)==0 else min(p_chck)[0]
            l_chck = [(i, i+len(tar)) for i in range(len(labl)-len(tar)+1) if labl[i:i+len(tar)] == tar]
            l_pick = 0 if len(l_chck)==0 else min(l_chck)[0]

            pred_a = nodes[p_pick].nloc[2]
            true_a = nodes[l_pick].nloc[2]
            z_tru_i=true_a
            z_prd_i=pred_a

        z_tru.append(z_tru_i)
        z_prd.append(z_prd_i)     
    
    pred_dict = {}
    pred_dict['x'] = x_loc
    pred_dict['y'] = y_loc
    pred_dict['z_true'] = z_tru
    pred_dict['z_pred'] = z_prd

    return(pd.DataFrame.from_dict(pred_dict))
