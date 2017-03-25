__author__ = 'zeboli1'
'''
Simple python scripts for visualize the voronoi diagram
'''
from PIL import Image
import random
import math
import lattice_info as lai

R_in = lai.R_in #A
R_buff = lai.R_buff #A
L = 2*R_buff
mult = 500/L



'''
input sites coordinates from generated sites informations
'''
#Inner discretized region
fpc = open('Inputs/Inner_sites.txt','r')
Cd_in = []
for line in fpc:
    Cd_in.append(map(float,line.split(' ')))
#Create Coordinates array
fpc.close()
N = len(Cd_in)

#Buffer region
fpbf = open('Inputs/Buffer_sites.txt','r')
Cd_bf = []
for line in fpbf:
    Cd_bf.append(map(float,line.split(' ')))
#Create Coordinates array
fpbf.close()
N_bf = len(Cd_bf)

Npt = N + N_bf

def generate_voronoi_diagram(width, height, num_cells):
	image = Image.new("RGB", (width, height))
	putpixel = image.putpixel
	imgx, imgy = image.size
	nx = []
	ny = []
	nr = []
	ng = []
	nb = []
	for i in range(N):
		nx.append((Cd_in[i][0]+R_buff)*mult)
		ny.append((Cd_in[i][1]+R_buff)*mult)
		nr.append(random.randrange(256))
		ng.append(random.randrange(256))
		nb.append(random.randrange(256))
	for i in range(N_bf):
		nx.append((Cd_bf[i][0]+R_buff)*mult)
		ny.append((Cd_bf[i][1]+R_buff)*mult)
		nr.append(random.randrange(256))
		ng.append(random.randrange(256))
		nb.append(random.randrange(256))
	for y in range(imgy):
		for x in range(imgx):
			dmin = math.hypot(imgx-1, imgy-1)
			j = -1
			for i in range(num_cells):
				d = math.hypot(nx[i]-x, ny[i]-y)
				if d < dmin:
					dmin = d
					j = i
			putpixel((x, y), (nr[j], ng[j], nb[j]))
	image.save("VoronoiDiagram.png", "PNG")
        image.show()

generate_voronoi_diagram(500, 500, Npt)
