import numpy as np
import pandas as pd
import string
from sklearn.preprocessing import LabelEncoder

def readTensor(file_name, data_path = ""):
    ft = pd.read_pickle(data_path + file_name)
    n_terms = ft.nunique().term
    n_authors = ft.nunique().author
    n_conf = ft.nunique().conf
      
    le = LabelEncoder()
    le.fit(list(set(ft.author)))
    ft['author_enc'] = le.transform(ft['author'])
    le.fit(list(set(ft.term)))
    ft['term_enc'] = le.transform(ft['term'])
    le.fit(list(set(ft.conf)))
    ft['conf_enc'] = le.transform(ft['conf'])
    print(ft.head())
    print(ft.shape)
    ft_array = np.array(ft)
    T = np.zeros((n_terms,n_authors,n_conf), dtype = np.int32)
    y = np.zeros(n_authors, dtype = np.int32)
    z = np.zeros(n_conf, dtype = np.int32)

    for i in range(ft_array.shape[0]):
        T[ft_array[i][6],ft_array[i][5],ft_array[i][7]] = 1
        y[ft_array[i][5]] = ft_array[i][3]
        z[ft_array[i][7]] = ft_array[i][4]
    return T, y, z, ft

if __name__ == '__main__':
    file_name = 'final.pkl'
    data_path = './resources/four_area/'

    T, y, z, ft = readTensor(file_name, data_path)
    '''
    target_file = open(data_path + 'target_reduced.txt', 'w+')
    d = T.shape
    #V_file = open(data_path + 'final.txt', 'w+')
    #for i in range(d[0]):
    #    for j in range(d[1]):
    #        for k in range(d[2]):                
    #            V_file.write(str(T[i,j,k]))
    for j in range(d[1]):                
        target_file.write(str(y[j]))
    target_file.write('\n')
    for k in range(d[2]):                
        target_file.write(str(z[k])) 

    #V_file.close()
    
    target_file.close()
    '''

