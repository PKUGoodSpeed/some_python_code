__author__ = 'zeboli1'

import numpy as np
import lattice_info as lai

b = lai.burg
Error = 1E-14
Thrs = 1E-4
LMax = 44.0*b

N_Cta = 10
Delta_Cta = 2.0*np.pi/N_Cta

#Getting data from the generated inner and buffer lattice
fpc = open('3D_FCC_sites/sites.txt','r')
sites = []
for line in fpc:
    sites.append(map(float,line.split(' ')))
fpc.close()
N_st = len(sites)
print N_st

geo_neibor = [[0 for i in xrange(N_st)] for i in xrange(N_st)]
vector = [[[0.0,0.0,0.0] for i in xrange(N_st)] for i in xrange(N_st)]
interface = [[0.0 for i in xrange(N_st)] for i in xrange(N_st)]
mid_pt = [[[0.0,0.0,0.0] for i in xrange(N_st)] for i in xrange(N_st)]

#Store mid points and vectors pointing from one site to another
for i in range(N_st):
    for j in range(N_st):
        mid_pt[i][j][0] = 0.5*(sites[i][0]+sites[j][0])
        mid_pt[i][j][1] = 0.5*(sites[i][1]+sites[j][1])
        mid_pt[i][j][2] = 0.5*(sites[i][2]+sites[j][2])
        if(i!=j):
            dx = sites[j][0] - sites[i][0]
            dy = sites[j][1] - sites[i][1]
            dz = sites[j][2] - sites[i][2]
            vector[i][j][0] = dx/np.sqrt(dx**2+dy**2+dz**2)
            vector[i][j][1] = dy/np.sqrt(dx**2+dy**2+dz**2)
            vector[i][j][2] = dz/np.sqrt(dx**2+dy**2+dz**2)
print "Generating vectors: DONE!"

#Getting geometrical neighboring information
for i in range(N_st):
    for j in range(i+1,N_st):
        geo_neibor[i][j]=1
        dx = sites[j][0] - mid_pt[i][j][0]
        dy = sites[j][1] - mid_pt[i][j][1]
        dz = sites[j][2] - mid_pt[i][j][2]
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        print i,j
        for k in range(N_st):
            dx1 =  sites[k][0] - mid_pt[i][j][0]
            dy1 =  sites[k][1] - mid_pt[i][j][1]
            dz1 =  sites[k][2] - mid_pt[i][j][2]
            dr1 = np.sqrt(dx1**2 + dy1**2 + dz1**2)
            if( (k!=i) & (k!=j) & (dr1 < dr)):
                geo_neibor[i][j] = 0
                break
        geo_neibor[j][i] = geo_neibor[i][j]

#voro_neighbor stores the indeces of neighbors to a certain site
voro_neighbor = [[] for i in xrange(N_st)]
for i in range(N_st):
    for j in range(i+1,N_st):
        if(geo_neibor[i][j]==1):
            voro_neighbor[i].append(j)
            voro_neighbor[j].append(i)
print "Generate Geometry neighbor information: DONE!"

#This function is to compute the minimum length from a certain point
#to any sites from voro_neighbor[i] and voroneighbor[j]
def MinLength(i,j,p):
    rm = LMax
    for k in voro_neighbor[i]:
        v = p - sites[k]
        dr= np.sqrt(np.dot(v,v))
        if((k!=j)&(k!=i)&(dr<rm)):
            rm = dr
    for k in voro_neighbor[j]:
        v = p - sites[k]
        dr= np.sqrt(np.dot(v,v))
        if((k!=i)&(k!=j)&(dr<rm)):
            rm = dr
    return(rm)

#Help to get the voronoi interface lengths
def Area(i,j):
    if(geo_neibor[i][j]==0):
        return(0.0)
    vx = vector[i][j][0]
    vy = vector[i][j][1]
    vz = vector[i][j][2]
    v0 = mid_pt[i][j]
    v1 = sites[i]
    if(vx>Thrs):
        x_axis = [-vy ,vx ,0.0]
        y_axis = [-vz ,0.0 ,vx]
    elif(vy>Thrs):
        x_axis = [vy, -vx, 0.0]
        y_axis = [0.0, -vz, vy]
    else:
        x_axis = [vz, 0.0 ,-vx]
        y_axis = [0.0, vz, -vy]
    x_axis = x_axis/np.sqrt(np.dot(x_axis,x_axis))
    y_axis = y_axis - np.dot(y_axis,x_axis)*x_axis
    y_axis = y_axis/np.sqrt(np.dot(x_axis,y_axis))
    d_Cta = Delta_Cta
    Cta = d_Cta/2.0
    area = 0.0
    while(Cta<2.0*np.pi):
        d_v = x_axis*np.cos(Cta) + y_axis*np.sin(Cta)
        a = 0.0
        b = LMax
        while((b-a)>Error):
            c = (b+a)*0.5
            v_p = v0 + c*d_v
            dr = np.sqrt(np.dot(v_p - v1,v_p - v1))
            drr = MinLength(i,j,v_p)
            if((drr >= dr)):
                a = c
            else:
                b = c
        area += 0.5*d_Cta*a**2
        Cta += d_Cta
    return area



#compute the voronoi interface lengths
for i in range(N_st):
    for j in voro_neighbor[i]:
        interface[i][j] = Area(i,j)
        interface[j][i] = interface[i][j]
        print i,j
print voro_neighbor[13]

print "Length of interfaces: DONE!"



#output data
'''
*****************
Two voronoi files:
1, voronoi_information (relationship between two sites i and j)
Format:
i   j   geo_neighbor[i][j]  interface[i][j]  vector[i][j]_x  vector[i][j]_y vector[i][j]_z

*****************
'''
fp = open('3D_FCC_sites/3D_voronoi_information.txt','w')
for i in range(N_st):
    for j in range(N_st):
        fp.write("%d"%i + ' ' + "%d"%j + ' ' + "%d"%geo_neibor[i][j] + ' ')
        fp.write("%.12f"%interface[i][j] + ' ')
        fp.write("%.12f"%vector[i][j][0] + ' ' + "%.12f"%vector[i][j][1] + ' ' + "%.12f"%vector[i][j][2])
        if((i!=N_st-1)|(j!=N_st-1)):
            fp.write('\n')
fp.close()
print "output Voronoi information: DONE!"

