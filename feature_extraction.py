import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline

#%% Inputs
# file directories
input_path = '---'
output_path = '---'

# samples
filenum = 1000 #number of files in the test case
angle = 180 # sample test case
run = 'Theta' + '%04d' % angle + 'deg'

# index column of considered parameter
ind_param = 4

#%% Plotting
def multiplot(x1, y1, sample, vmin=None, vmax=None,
              cmap='jet', D=1, title='', sp=[]):
    """
    Plotting a 2D contour using three 1D arrays
    x1: horizontal axis coordinates
    y1: vertical axis coordinates
    sample: contour values
    D: reference value for nondimensionalization
    sp: subplot position as [nrows, ncols, index]
    """
    x1 = x1/D
    y1 = (y1 - max(y1)/2)/D
    
    val = np.reshape(sample, (len(x1),len(y1)))
    val = np.transpose(val)
    
    if not sp:
        ax = plt.axes(aspect='equal')
        plt.contourf(x1, y1, val, levels=101, vmin=vmin, vmax=vmax, cmap=cmap)
    else :
        ax = plt.subplot(sp[0],sp[1],sp[2], aspect='equal')
        plt.contourf(x1, y1, val, levels=101, vmin=vmin, vmax=vmax, cmap=cmap)
    
    # Geometry of tested model as a circular cylidner
    cylinder = mpatches.CirclePolygon((0,0), radius=0.5, resolution=50, 
                               edgecolor='k', facecolor='w', linewidth=1.2)
    ax.add_patch(cylinder)
    ax.tick_params(labelsize=19)
    
    textprop = {'fontsize':22, 'fontstyle':'italic', 'fontname':'serif'}
    plt.xlim([-0.5,4.8])
    plt.ylim([-1.6,1.6])
    plt.xlabel('x*', textprop)
    plt.ylabel('y*', textprop)
    plt.title(title, textprop)
    
#%% reading files
filename = '%s\\%s%06dn.dat' % (input_path, run, 1)

allparam = np.loadtxt(filename, delimiter=' ', skiprows=1, max_rows=2000)
x = np.unique(allparam[:,0])
y = np.unique(allparam[:,1])

ncol = len(x)
nrow = len(y)
data = np.empty([filenum,nrow*ncol])

for i in range(filenum):
    filename = filename.replace('%06d' % i,'%06d' % (i+1))
    data[i,:] = np.loadtxt(filename, delimiter=' ', 
                            skiprows=1, usecols = ind_param)

#%% Exploring Data
columns = ['X mm', 'Y mm', 'U mm/s', 'V mm/s', 'Vorticity 1/s', \
           'Velocity Magnitude mm/s', 'TKE (mm/s)^2', 'TKErms (mm/s)^2']
allparam_df=pd.DataFrame(allparam, columns=columns)
print(allparam_df.info())

plt.figure()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
for i in range(6):
    multiplot(y, x, data[i,:], vmin=-46, vmax=46,
              title='Instantaneous Vorticity %d' % (i+1), sp=[2,3,i+1])

#%% Feature Extraction

# scaling
scaler = MinMaxScaler()

# NMF
n_components = 6
model = NMF(n_components)

# Pipeline
pipeline = Pipeline([('scaler',scaler),('NMF',model)])
pipeline.fit(data)
pipeline.transform(data)

plt.figure()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
for i, component in enumerate(normalize(model.components_,axis=0), start=1):
    multiplot(y, x, component, title='NMF Component %d' % i, 
              sp=[2,n_components/2,i])

#%% Writing Data
outputfile = '%s\\%s_NMF_components.dat' % (output_path, run)

header = 'TITLE= "%s" VARIABLES="x (mm)", "y (mm)", "Component 1"' % run
for i in range(model.n_components_-1):
    header = header + ', "Component %d"' % (i+2)
header =  header + ' ZONE I=73, J=99, F=POINT \n'

output = np.transpose(np.vstack((allparam[:,0],allparam[:,1],model.components_)))
with open(outputfile,"w") as file1:
    file1.write(header)
    np.savetxt(file1,output, fmt='%0.6f')
