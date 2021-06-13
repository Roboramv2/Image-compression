import os
from matplotlib import pyplot as plt

#indices
n = ['1', '2', '3', '4', '5']
m = ['1', '2', '3']

#generating filenames
fils = []
for i in n:
    for j in m:
        name = i+'_'+j
        fils.append(name)

#reading files in directory
sizes = {}
filnames = os.listdir('.')
for i in fils:
    for j in filnames:
        if (i == j.split('.')[0]):
            size = os.path.getsize(j) 
            sizes[i]=size

#categorising the sizes
original = []
vector = []
final = []
for i in sizes:
    if i[2]=='1':
        original.append(sizes[i])
    elif i[2]=='2':
        vector.append(sizes[i])
    elif i[2]=='3':
        final.append(sizes[i])

#plotting
plt.plot(vector, original, color='#00e1ff', linewidth=2, label='compressed')
plt.plot(final, original, color='#10e1ff', linewidth=2, label='decompressed')
