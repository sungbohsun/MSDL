import os
import msaf
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from glob import glob
from multiprocessing import Pool

def BD(name):
        
    out_name = 'label/'+name[:-4]+'.txt'

    try:
        boundaries, labels = msaf.process('audio_rename/'+name, 
                                          boundaries_id='sf',
                                          feature='mfcc',
                                          #labels_id='cnmf', 
                                          plot = False)
        m = list(boundaries)
        sec = 3
        for c in m:
            [m.remove(L) for L in np.array(m)[(abs(m-c)<sec)&(0<abs(m-c))]]
        c = list(map(int,m))
        f = open(out_name, mode='w')
        for i in range(len(c)-1):
            f.write('{} {} N\n'.format(c[i],c[i+1]-1))
            f.write('{} {} T\n'.format(c[i+1]-1,c[i+1]))
        f.close()
        print('-'*8+'success',out_name)
    except:
        print('-'*8+'error',out_name)

if __name__ == '__main__':  
    
    rate = 22050
        
    if not os.path.isdir('label'):
        os.mkdir('label')
        
        
    names_ = [file.split('/')[1][:-4]+'.mp3' for file in sorted(glob('audio_rename/*'))]
    names= [c for c in names_ if not os.path.isfile('label/'+c[:-4]+'.txt')]
    print('have {} data {} need proccess'.format(len(names_),len(names)))
    
    with Pool(5) as pool:  
        result = pool.map(BD,names)