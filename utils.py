import sys
import os
import numpy as np
import logging as l
import datetime
from algorithms.coclust_3D_tau import CoClust
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari


def execute_test(f, V, x, y, z, noise=0, algorithm = 'ALT2'):
    '''
    Execute CoClust algorithm and write an output file (already existing and open)

    Parameters:
    ----------

    f: output file (open). See CreateOutputFile for a description of the fields.      
    V: tensor
    x: target on mode 0
    y: target on mode 1
    z: target on mode 2
    noise: only for synthetic tensors. Amount of noise added to the perfect tensor
    algorithm: optimization strategy. One of {'ALT', 'AVG', 'AGG', 'ALT2', 'AGG2'}
    sparsity: sparsity of the tensor (number of entries != 0 / total number of entries)

    '''
    
    model = CoClust(np.sum(V.shape) * 100, optimization_strategy = algorithm)
    model.fit(V)
    nmi_x = nmi(x, model.x_, average_method='arithmetic')
    nmi_y = nmi(y, model.y_, average_method='arithmetic')
    nmi_z = nmi(z, model.z_, average_method='arithmetic')
    ari_x = ari(x, model.x_)
    ari_y = ari(y, model.y_)
    ari_z = ari(z, model.z_)
    tau = model.final_tau_

    sparsity = 1 - (np.sum(V>0) / np.product(V.shape))

    f.write(f"{V.shape[0]},{V.shape[1]},{V.shape[2]},{np.max(x) + 1},{np.max(y) + 1},{np.max(z) + 1},{noise},{tau[0]},{tau[1]},{tau[2]},{nmi_x},{nmi_y},{nmi_z},{ari_x},{ari_y},{ari_z},{model._n_clusters[0]},{model._n_clusters[1]},{model._n_clusters[2]},{model.execution_time_},{sparsity},{algorithm}\n")


def CreateOutputFile(partial_name, own_directory = False, date = True, overwrite = False):
    '''
    Create and open a file containing the header described below.

    Parameters:
    ----------
    partial_name: partial name of the file and the directory that will contain the file.
    own_directory: boolean. Default: False.
        If true, a new directory './output/_{partial_name}/aaaa-mm-gg_hh.mm.ss' will be created.
        If flase, the path of the file will be './output/_{partial_name}'.
    date: boolean. Default: True.
        If true, the file name will include datetime.
        If false, it will not.
                

    Output
    ------
    f: file (open). Each record contains the following fields, separated by commas (csv file):
        - dim_x: dimension of the tensor on mode 0
        - dim_y: dimension of the tensor on mode 1
        - dim_z: dimension of the tensor on mode 2
        - x_num_classes: correct number of clusters on mode 0
        - y_num_classes: correct number of clusters on mode 1
        - z_num_classes: correct number of clusters on mode 2
        - noise: only for synthetic tensors. Amount of noise added to the perfect tensor
        - tau_x: final tau_{x|y,z}
        - tau_y: final tau_{y|x,z}
        - tau_z: final tau_{z|x,y}
        - nmi_x: normalized mutual information score on mode 0
        - nmi_y: normalized mutual information score on mode 1
        - nmi_z: normalized mutual information score on mode 2
        - ari_x: adjusted rand index on mode 0
        - ari_y: adjusted rand index on mode 1
        - ari_z: adjusted rand index on mode 2
        - x_num_clusters: number of clusters on mode 0 detected by CoClust
        - y_num_clusters: number of clusters on mode 1 detected by CoClust
        - z_num_clusters: number of clusters on mode 2 detected by CoClust
        - execution time
        - sparsity: sparsity of the tensor (number of entries != 0 / total number of entries)
        - algorithm: optimization strategy. One of {'ALT', 'AVG', 'AGG', 'ALT2', 'AGG2'}

        File name:{partial_name}_aaaa-mm-gg_hh.mm.ss.csv or {partial_name}_results.csv
    dt: datetime (as in the directory/ file name)

    
    '''

    
    dt = f"{datetime.datetime.now()}"
    if own_directory:
        data_path = f"./output/_{partial_name}/" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + "/"
    else:
        data_path = f"./output/_{partial_name}/"
    directory = os.path.dirname(data_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    new = True
    if date:
        file_name = partial_name + "_" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + ".csv"
    else:
        file_name = partial_name + '_results.csv'
        if os.path.isfile(data_path + file_name):
            if overwrite:
                os.remove(data_path + file_name)
            else:
                new = False
            
            
    f = open(data_path + file_name, "a",1)
    if new:
        f.write("dim_x,dim_y,dim_z,x_num_classes,y_num_classes,z_num_classes,noise,tau_x,tau_y,tau_z,nmi_x,nmi_y,nmi_z,ari_x,ari_y,ari_z,x_num_clusters,y_num_clusters,z_num_clusters,execution_time,sparsity,algorithm\n")

    return f, dt


def CreateLogger(input_level = 'INFO'):
    level = {'DEBUG':l.DEBUG, 'INFO':l.INFO, 'WARNING':l.WARNING, 'ERROR':l.ERROR, 'CRITICAL':l.CRITICAL}
    logger = l.getLogger()
    logger.setLevel(level[input_level])

    return logger
