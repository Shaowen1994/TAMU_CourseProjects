import os

path = '../Datasets/For_Seq_Stru/features_ss/'
new_path = '../Datasets/For_Seq_Stru/features/'

file_list = os.listdir(path)

for f in file_list:

    print(f)

    with open(path + f,'r') as f_read:
        lines = f_read.readlines()
    with open(new_path + f.strip('_ss'),'w') as f_write:
        f_write.write(lines[0])
        feature = ''
        info = lines[1].split('\t')
        feature += info[0] + '\t'
 
        one_hot = [i for i in info[1].split(' ') if i != '']

        num = 0
        counter = 0

        for pair in one_hot:
            index = int(pair.split(':')[0])
            counter += 1
            if counter == 23:
                counter = 0
            elif counter <= 20:            
                num += 1
                feature += str(num) + ':' + pair.split(':')[1] + ' '
        f_write.write(feature)
        print(num/20,' ',len(one_hot)/23)


