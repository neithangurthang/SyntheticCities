import pandas as pd
import numpy as np
import geopandas as gpd
import fiona
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib as mpl

src = "../../data/zg_av_lv95.gpkg"
fiona.listlayers(src)

zur = gpd.read_file("../../data/zh_av_lv95.gpkg", layer='lcsf')
zug = gpd.read_file(src, layer='lcsf')

probMap = {'edificio':'b', 
           'altro_rivestimento_duro':'r', 
           'giardino':'g', 
           'marciapiede':'r',
           'campo_prato_pascolo':'g',
           'altro_humus':'g',
           'vigna':'g',
           'ferrovia': 'g',
           'bosco_fitto': 'g', 
           'bacino_idrico': 'g', 
           'spartitraffico':'r', 
           'corso_acqua':'g',
           'specchio_acqua':'g', 
           'strada_sentiero':'r',
           'pietraia_sabbia':'g',
           'torbiera': 'g',
           'altra_coltura_intensiva': 'g',
           'altro_bosco': 'g',
           'cava_di_ghiaia_discarica': 'g',
           'pascolo_boscato_fitto': 'g',
           'canneti': 'g',
           'roccia': 'g',
           'bacino_idrico': 'g',
           'specchio_acqua': 'g'
          }

def cut_and_print_probs(geometries, step, path):
    minX, minY, maxX, maxY = geometries.total_bounds
    repsX, repsY= int((maxX-minX)/step), int((maxY-minY)/step)
    for x in range(repsX):#range(1):
        for y in range(repsY): #range(1):
            box = gpd.GeoSeries([Polygon([(minX + (x+0) * step, minY + (y+0) * step),
                                         (minX + (x+1) * step, minY + (y+0) * step),
                                         (minX + (x+1) * step, minY + (y+1) * step),
                                         (minX + (x+0) * step, minY + (y+1) * step)])
                                      ],)
            boxGDF = gpd.GeoDataFrame.from_features(box)
            boxGDF = boxGDF.set_crs('EPSG:2056')
            res = gpd.sjoin(geometries, boxGDF, how='inner', predicate='intersects')
            res['col']= res['Genere'].map(probMap)
            nObjs = res.shape[0]
            nBld = res[res.Genere == 'edificio'].shape[0]
            
            # PLOT
            resolution = 72 # in dpi
            plt.ioff() # turns off interactive plotting
            if (nObjs > 100) and (nBld>10): # only interesting plots
                filename = path + 'CadastralX' + str(int(minX+x*step)) + 'Y' + str(int(minY+y*step)) + 'S' + str(step) + ".png" 
                #larger size, to use the crop in the dataloader
                res_multiplier = 20
                fig, ax = plt.subplots(figsize=(res_multiplier*64/resolution,res_multiplier*64/resolution), dpi=resolution, frameon=False)
                # ax = plt.axes([0,0,1,1], frameon=False)
                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
                # plt.autoscale(tight=True)
                ax.axis('off')
                mpl.rcParams['savefig.pad_inches'] = 0
                
                ax.set_xlim(boxGDF.bounds['minx'].item(), boxGDF.bounds['maxx'].item())
                ax.set_ylim(boxGDF.bounds['miny'].item(), boxGDF.bounds['maxy'].item())
                res.plot(ax=ax, color=res['col'])
                plt.savefig(filename)
                plt.close(fig)

    return

pathRGB = '../../../cadastralExportRGB/'

cut_and_print_probs(zur, 250, pathRGB)
cut_and_print_probs(zug, 250, path=pathRGB)