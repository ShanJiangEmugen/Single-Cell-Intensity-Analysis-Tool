import os
import sys
import pandas as pd
from aicsimageio import AICSImage
from skimage import measure
from cellpose import models
import matplotlib.pyplot as plt
import numpy as np
import yaml

import warnings
warnings.filterwarnings("ignore")

def calculate_median(regionmask, intensity):
    return np.median(intensity[regionmask])
    
def single_image(image_path, parameters=None):
    '''
    Note: path here should only contain the file name, not the full path
    
    The output .csv will be save in same address as the input image
    '''
    
    diameter = parameters['diameter']
    flow_threshold = parameters['flow_threshold']
    cellprob_threshold = parameters['cellprob_threshold']
    min_size = parameters['min_size']
    
    img = AICSImage(image_path)
    data_czyx_ori = img.get_image_data("YXZ", C=0, S=0, T=0)  # returns 3D ZYX numpy array
    data_czyx_grey = img.get_image_data("YXZ", C=1, S=0, T=0)  # returns 3D ZYX numpy array

    model = models.Cellpose(model_type='cyto', gpu=True)

    channels = [[0,0]] # seems not a big deal in our case
    masks, flows, styles, diams = model.eval(data_czyx_grey, diameter=diameter, channels=channels,
                                             flow_threshold=flow_threshold, 
                                             cellprob_threshold=cellprob_threshold,
                                             min_size=min_size
                                            )


    props = measure.regionprops_table(masks, data_czyx_ori, separator='_', properties=['label',
                                                                                       'centroid',
                                                                                       'area',
                                                                                       'mean_intensity',],
                                     extra_properties=(calculate_median,))
    props_df = pd.DataFrame(props)

    # save dataframe
    return props_df, masks, data_czyx_grey, data_czyx_ori

def save_figures(full_path, masks, data_czyx_grey, data_czyx_ori, base):
    # save figure here
    masks[masks==0] = 999
    
    fig = plt.figure(figsize=[16,14])
    plt.subplot(221)
    plt.imshow(data_czyx_grey, cmap='Reds')
    plt.title('Grey Scale')
    plt.subplot(222)
    plt.title('Grey Scale + Mask')
    plt.imshow(data_czyx_grey, cmap='Reds')
    plt.imshow(masks,alpha=0.3*(data_czyx_grey>0)[:,:,0], cmap='Blues_r')
    
    # also include original
    plt.subplot(223)
    plt.imshow(data_czyx_ori, cmap='Reds_r')
    plt.title('Intensity')
    plt.subplot(224)
    plt.title('Intensity + Mask')
    plt.imshow(data_czyx_ori, cmap='Reds_r')
    plt.imshow(masks,alpha=0.3*(data_czyx_grey>0)[:,:,0], cmap='Greens')
    
    plt.tight_layout()
    fig.savefig('{}/{}.png'.format(os.path.dirname(full_path), base))
    plt.close()
    
if __name__ == '__main__':
    full_path = sys.argv[1]
    image_path = full_path.split('/')[-1]
    
    with open('config.yaml') as f:
    	param = yaml.load(f, Loader=yaml.FullLoader)
    
    props_df, masks, data_czyx_grey, data_czyx_ori = single_image(full_path, param)
    
    base = os.path.splitext(image_path)[0]
    props_df.to_csv('{}/{}.csv'.format(os.path.dirname(full_path), base))
    
    # save figure here 
    masks[masks==0] = 999

    fig = plt.figure(figsize=[16,14])
    plt.subplot(221)
    plt.imshow(data_czyx_grey, cmap='Reds')
    plt.title('Grey Scale')
    plt.subplot(222)
    plt.title('Grey Scale + Mask')
    plt.imshow(data_czyx_grey, cmap='Reds')
    plt.imshow(masks,alpha=0.3*(data_czyx_grey>0)[:,:,0], cmap='Blues_r')
        
    # also include original
    plt.subplot(223)
    plt.imshow(data_czyx_ori, cmap='Reds_r')
    plt.title('Intensity')
    plt.subplot(224)
    plt.title('Intensity + Mask')
    plt.imshow(data_czyx_ori, cmap='Reds_r')
    plt.imshow(masks,alpha=0.3*(data_czyx_grey>0)[:,:,0], cmap='Greens')
    
    plt.tight_layout()
    fig.savefig('{}/{}.png'.format(os.path.dirname(full_path), base))
    plt.close()
