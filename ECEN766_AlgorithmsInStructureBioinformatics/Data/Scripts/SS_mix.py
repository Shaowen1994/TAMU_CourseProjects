##################################################################################################
# Mix the real SS features and the predicted ones together according to the input real portion.
##################################################################################################

import sys 
import os
import numpy as np

real_portion = int(sys.argv[1])

targert_path = '../Datasets/For_Seq_Stru/features_mix_%d/'%real_portion

if not os.path.exists(targert_path):
    print('Error! %s does not exist!'%targert_path)
    quit()

real_path = '../Datasets/For_Seq_Stru/features_ss/' 
pre_path = '../Datasets/For_Seq_Stru/features_pre/'

f_list = os.listdir(real_path)
l = len(f_list)

select = np.random.choice(l,int(l*real_portion/100),replace=False)

f_r = open(targert_path + 'real_features','w')
f_p = open(targert_path + 'predicted_features','w')

for i in range(l):
    if i in select:
        os.system('cp %s%s %s'%(real_path,f_list[i],targert_path))
        f_r.write(f_list[i] + '\n')
    else:
        os.system('cp %s%s %s'%(pre_path,f_list[i],targert_path))
        f_p.write(f_list[i] + '\n')

f_r.close()
f_p.close()
