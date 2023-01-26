import random
import verde as vd
import numpy as np
from .import_classes import Station


def generate_synth_stations(griddf, num_stations, vert_res=30, feat_type='A'):

    # for finding basement value

    spline = vd.Spline(damping=0.0008)
    spline.fit((griddf['x'],griddf['y']), griddf['z'])

    x_array = [random.randint(griddf.x.min(),griddf.x.max()) for _ in range(num_stations)]
    y_array = [random.randint(griddf.y.min(),griddf.y.max()) for _ in range(num_stations)]

    station_list = []

    station_z = np.linspace(0,griddf.z.min(),vert_res)

    for i in range(num_stations):

        # TODO you would sample the basement location from the grid based on randomly generated xy
        basement_depth = spline.predict((float(x_array[i]),float(y_array[i]),0))

        points_above = [x for x in station_z if x>basement_depth]
        points_below = [x for x in station_z if x<=basement_depth]

        # feature_type A
        if feat_type=='A':
            feats_above = [np.sin(x) for x in np.linspace(0,3,len(points_above))]
            feats_below = [(np.sin(x)*-1)/2 for x in np.linspace(0,3,len(points_below))]

        # feature_type B
        if feat_type=='B':
            feats_above = list(np.random.choice(np.linspace(-0.02, 0.02,100), len(points_above), replace=False))
            feats_below = list(np.random.choice(np.linspace(-1, 1, 100), len(points_below), replace=False))
        
        loc = [x_array[i],y_array[i],basement_depth]
        params = {'depth':station_z,'rdata':feats_above+feats_below,'label':[0]*len(points_above)+[1]*len(points_below)}

        station_list.append(Station(i,loc,feat_type,params))


    return(station_list)