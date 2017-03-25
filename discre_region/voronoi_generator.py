__author__ = 'zeboli1'

import numpy as np
import lattice_info as lai

b = lai.burg
Error = 1E-14
LMax = 16.0*b

#Getting data from the generated inner and buffer lattice
fpc = open('Inputs/Inner_sites.txt','r')
sites = []
for line in fpc:
    sites.append(map(float,line.split(' ')))
fpc.close()
N_in = len(sites)
print N_in

fpc = open('Inputs/Buffer_sites.txt','r')
for line in fpc:
    sites.append(map(float,line.split(' ')))
fpc.close()
N_tt = len(sites)
print "Extract sites from inner and buffer regions: DONE!"

geo_neibor = [[0 for i in xrange(N_tt)] for i in xrange(N_tt)]
vector = [[[0.0,0.0] for i in xrange(N_tt)] for i in xrange(N_tt)]
interface = [[0.0 for i in xrange(N_tt)] for i in xrange(N_tt)]
mid_pt = [[[0.0,0.0] for i in xrange(N_tt)] for i in xrange(N_tt)]
Volm = [0.0 for i in xrange(N_tt)]

#Store mid points and vectors pointing from one site to another
for i in range(N_tt):
    for j in range(N_tt):
        mid_pt[i][j][0] = 0.5*(sites[i][0]+sites[j][0])
        mid_pt[i][j][1] = 0.5*(sites[i][1]+sites[j][1])
        if(i!=j):
            dx = sites[j][0] - sites[i][0]
            dy = sites[j][1] - sites[i][1]
            vector[i][j][0] = dx/np.sqrt(dx**2+dy**2)
            vector[i][j][1] = dy/np.sqrt(dx**2+dy**2)

#Getting geometrical neighboring information
for i in range(N_tt):
    for j in range(i+1,N_tt):
        geo_neibor[i][j]=1
        dx = sites[j][0] - mid_pt[i][j][0]
        dy = sites[j][1] - mid_pt[i][j][1]
        dr = np.sqrt(dx**2 + dy**2)
        for k in range(N_tt):
            dx1 =  sites[k][0] - mid_pt[i][j][0]
            dy1 =  sites[k][1] - mid_pt[i][j][1]
            dr1 = np.sqrt(dx1**2 + dy1**2)
            if( (k!=i) & (k!=j) & (dr1 < dr)):
                geo_neibor[i][j] = 0
                break
        geo_neibor[j][i] = geo_neibor[i][j]

#voro_neighbor stores the indeces of neighbors to a certain site
voro_neighbor = [[] for i in xrange(N_tt)]
for i in range(N_tt):
    for j in range(i+1,N_tt):
        if(geo_neibor[i][j]==1):
            voro_neighbor[i].append(j)
            voro_neighbor[j].append(i)
print "Generate Geometry neighbor information: DONE!"

#This function is to compute the minimum length from a certain point
#to any sites from voro_neighbor[i] and voroneighbor[j]
def MinLength(i,j,x,y):
    rm = LMax
    for k in voro_neighbor[i]:
        dx = x - sites[k][0]
        dy = y - sites[k][1]
        dr= np.sqrt(dx**2 + dy**2)
        if((k!=j)&(k!=i)&(dr<rm)):
            rm = dr
    for k in voro_neighbor[j]:
        dx = x - sites[k][0]
        dy = y - sites[k][1]
        dr= np.sqrt(dx**2 + dy**2)
        if((k!=i)&(k!=j)&(dr<rm)):
            rm = dr
    return(rm)

#Help to get the voronoi interface lengths
def Searching(i,j,hj):
    if(geo_neibor[i][j]==0):
        return(0.0)
    a = 0.0
    b = LMax
    xi = sites[i][0]
    yi = sites[i][1]
    xm = mid_pt[i][j][0]
    ym = mid_pt[i][j][1]
    vx = vector[i][j][1]*hj
    vy = -vector[i][j][0]*hj
    while((b-a)>Error):
        c = (b+a)*0.5
        xc = xm + vx*c
        yc = ym + vy*c
        dr = (xc-xi)**2 + (yc-yi)**2
        drr = MinLength(i,j,xc,yc)
        if((drr >= dr)):
            a = c
        else:
            b = c
    return(a)

#compute the voronoi interface lengths
for i in range(N_tt):
    for j in voro_neighbor[i]:
        interface[i][j] = Searching(i,j,1.0)+Searching(i,j,-1.0);
        interface[j][i] = interface[i][j]
print voro_neighbor[13]

print "Length of interfaces: DONE!"

#compute the voronoi area for each site
for i in range(N_tt):
    Volm[i]=0.0
    for j in voro_neighbor[i]:
        dx = sites[i][0] - sites[j][0]
        dy = sites[i][1] - sites[j][1]
        dr = np.sqrt(dx**2+dy**2)
        Volm[i] += 0.25*dr*interface[i][j]

print "Voronoi Area for each site: DONE!"


#output data
'''
*****************
Two voronoi files:
1, voronoi_information (relationship between two sites i and j)
Format:
i   j   geo_neighbor[i][j]  interface[i][j]  vector[i][j]_x  vector[i][j]_y

2, voronoi_area (voronoi area for each site)
Format:
i   area[i]
*****************
'''
fp = open('Inputs/voronoi_information.txt','w')
for i in range(N_tt):
    for j in range(N_tt):
        fp.write("%d"%i + ' ' + "%d"%j + ' ' + "%d"%geo_neibor[i][j] + ' ')
        fp.write("%.12f"%interface[i][j] + ' ')
        fp.write("%.12f"%vector[i][j][0] + ' ' + "%.12f"%vector[i][j][1])
        if((i!=N_tt-1)|(j!=N_tt-1)):
            fp.write('\n')
fp.close()

fp = open('Inputs/voronoi_area.txt','w')
for i in range(N_tt):
    fp.write("%d"%i + ' ' "%.12f"%Volm[i])
    if(i!=N_tt-1):
        fp.write('\n')
fp.close()

