import os
import sys
import pandas as pd
from aicsimageio import AICSImage
from skimage import measure
from cellpose import models
from cellpose import plot
from tqdm import tqdm
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import yaml
from scipy.ndimage import median_filter # for despeckle
import scikit_posthocs as sp
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings("ignore")


def calculate_median(regionmask, intensity):
    return np.median(intensity[regionmask])


def make_dir(path):
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create folder
    folder_name = 'intensity_result_{}'.format(timestamp)
    os.makedirs(os.path.join(path,folder_name))

    return folder_name
    
def batch_image(folder, param=None, cell_size_threshold=None, group_info=None, 
                gamma=False, extra_channel=False, mask_ch=1, signal_ch=0):
    '''
    Note: this function will create a folder that hold all results
    '''
    
    diameter = param['diameter']
    flow_threshold = param['flow_threshold']
    cellprob_threshold = param['cellprob_threshold']
    min_size = param['min_size']
    
    # see the entire dataset
    # get all files 
    img_fns = [f for f in os.listdir(folder) if not f.startswith('.') and f.endswith('czi')]
    #img_fns.sort(key=lambda x:int(x.split('_')[0].split(' ')[1]))
    img_fns.sort(key=lambda x:(int(x.split('_')[0]),int(x.split('_')[8]), int(x.split('_')[9])))
    folder_name = make_dir(folder)
    print(img_fns)
    print(folder_name)
    # initilize model
    model = models.Cellpose(model_type='cyto', gpu=True)
    channels = [[0, 0]]
    
    master_df = pd.DataFrame()
    col_names = []
    # load 1 file at time # 
    for f in tqdm(range(len(img_fns)), desc='Cellpose analyzing progress'):
        img = AICSImage(os.path.join(folder, img_fns[f]))
        data_czyx_ori = img.get_image_data("YX", C=0, S=0, T=0)  # returns 3D ZYX numpy array
        data_czyx_grey = img.get_image_data("YX", C=1, S=0, T=0)  # returns 3D ZYX numpy array
        
        # despeckle here
        data_czyx_ori = median_filter(data_czyx_ori, size=3)  # the default size in ImageJ is 3x3
        data_czyx_grey = median_filter(data_czyx_grey, size=3) # this step will takes about 2 seconds for each image(2048x2048)
        
        if extra_channel:
            data_czyx_3 = img.get_image_data("YX", C=2, S=0, T=0)  # returns 3D ZYX numpy array
            data_czyx_3 =  median_filter(data_czyx_3, size=3)
        
        else: 
            data_czyx_3 = None
        
        # cellpose get masks here
        channels = [[2,1]] # seems not a big deal in our case
        
        mask_channel = [data_czyx_ori, data_czyx_grey, data_czyx_3][mask_ch] # default was mask on grey channel
        signal_channel = [data_czyx_ori, data_czyx_grey, data_czyx_3][signal_ch] # default was mask on red channel
        
        masks, flows, styles, diams = model.eval(mask_channel, diameter=diameter, channels=channels,
                                                 flow_threshold=flow_threshold,
                                                 cellprob_threshold=cellprob_threshold,
                                                 min_size=min_size
                                                )
        # save some info
        if gamma:
            img_norm1 = signal_channel/255.0  #注意255.0得采用浮点数
            img_gamma = np.power(img_norm1,gamma)*255.0
            props = measure.regionprops_table(masks, img_gamma, separator='_', 
                                              properties=['label',
                                                          'centroid',
                                                          'area',
                                                          'mean_intensity'],
                                             extra_properties=(calculate_median,))
        else:
            
            props = measure.regionprops_table(masks, signal_channel, separator='_', 
                                              properties=['label',
                                                          'centroid',
                                                          'area',
                                                          'mean_intensity'],
                                             extra_properties=(calculate_median,))
        props_df = pd.DataFrame(props)

        # export to .csv file
        base = os.path.splitext(img_fns[f])[0]
        props_df.to_csv('{}/{}/{}.csv'.format(folder, folder_name, base))
        
        # save the df for later use
        col_names.append(base)
        valid_cells = props_df[props_df['area'] >= cell_size_threshold]
        master_df = pd.concat([master_df, valid_cells['calculate_median']], axis=1)
        master_df.columns = col_names
        
        # save figure here 
        # remove smaller cells from the mask
        temp_msk = masks.copy()
        temp_msk[temp_msk!=0] = 1
        small_cells = props_df[props_df['area'] < cell_size_threshold]['label'].values

        for cell in small_cells:
          temp_msk[temp_msk==cell]=9999
        
        fig = plt.figure(figsize=[21,10])
        plt.subplot(241)
        plt.imshow(data_czyx_grey, cmap='Reds')
        plt.title('Grey Scale')
        plt.subplot(242)
        plt.title('Masks')
        plt.imshow(data_czyx_grey, cmap='Reds')
        plt.imshow(temp_msk,cmap='Reds_r')
                
        mask_plot = make_mask_overlay(data_czyx_grey, masks)
        outline_plot = plot.outline_view(data_czyx_grey[:,:],  masks, color=[0,255,0], mode='thick')
        plt.subplot(243)
        plt.imshow(mask_plot[:,:,0], cmap=make_cmap())
        plt.title('Masks')
        
        plt.subplot(244)
        plt.imshow(outline_plot.astype(np.uint8))
        plt.title('Grey Scale + Outline')
        
        # also include original
        plt.subplot(245)
        plt.imshow(data_czyx_ori, cmap='Reds_r')
        plt.title('Intensity')
        plt.subplot(246)
        plt.title('Intensity + Mask')
        plt.imshow(data_czyx_ori*temp_msk, cmap='Reds_r')
                
        # add more plot here
        mask_plot = plot.mask_overlay(data_czyx_ori[:,:], masks)
        outline_plot = plot.outline_view(data_czyx_ori[:,:],  masks, color=[0,255,0], mode='thick')
        plt.subplot(247)
        plt.imshow(mask_plot, alpha=1)
        plt.title('Intensity + Mask Overlay')
        
        plt.subplot(248)
        plt.imshow(outline_plot.astype(np.uint8))
        plt.title('Intensity + Outline')

        plt.tight_layout()
        fig.savefig('{}/{}/{}.png'.format(folder, folder_name, base))
        plt.close()
        
    
    # sort the master df by input group_info first
    # also make a stacked result:

    output_master = pd.DataFrame()
    stacked_master_data={}
    col_names = []
    for key, indces in group_info.items():
        temp_stack = pd.DataFrame()
        indces = np.array(indces)-1
        current_data = master_df.iloc[:,indces]
        output_master = pd.concat([output_master, current_data], axis=1)
        for i in range(len(indces)):
            # make a column name first
            col_names.append('{}_{}'.format(key, indces[i]+1))
            # then stack all data in 1 column for stacked master
            temp_stack = pd.concat([temp_stack, current_data.iloc[:,i].dropna()], ignore_index=True, axis=0)
        
        stacked_master_data[key] = temp_stack.values.flatten()
    output_master.columns = col_names
    output_master.to_csv('{}/{}/sorted_master.csv'.format(folder, folder_name))
    stacked_master = pd.DataFrame(stacked_master_data.values()).transpose()
    stacked_master.columns=stacked_master_data.keys()
    stacked_master.to_csv('{}/{}/stacked_master.csv'.format(folder, folder_name))
    
def make_individual_plot(path, group_info, color_lst, 
                         measurments = 'median', # or 'mean'
                         scatter=50, scatter_size=5,
                         width=0.5, font=12,
                         figsize=[6,4]):
    # load csv file data
    if measurments == 'median':
        sorted_results = pd.read_csv(os.path.join(path, 'median_sorted_master.csv'), index_col=0)
    elif measurments == 'mean':
        sorted_results = pd.read_csv(os.path.join(path, 'mean_sorted_master.csv'), index_col=0)
    else:
        print('check your measurments input for the coming error!')
        
    idx_counter = 0
    pos_counter = 0
    cidx = 0
    xtick_idx = []
    viridis = cm.get_cmap('hsv', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    fig = plt.figure(figsize=figsize)
    for group in group_info.values():
      for i in range(len(group)):
        temp_col = sorted_results.columns[idx_counter]
        current_img = sorted_results[temp_col].dropna()
        violin = plt.violinplot(current_img, positions=[pos_counter],
                          showextrema=False, showmeans=False, widths=[width],
                          showmedians=True, quantiles=[0.25,0.75]
                          )
        for pc in violin['bodies']:
          pc.set_facecolor(color_lst[cidx])
          pc.set_edgecolor('black')
          pc.set_alpha(0.2)
        vp= violin['cmedians']
        vp.set_edgecolor('black')
        vp.set_linewidth(2)
        vp = violin['cquantiles']
        vp.set_edgecolor('blue')
        vp.set_linewidth(1)
    
        # also makes some scatter here
        if scatter:
            # check if there is enough points for scattering
            if len(current_img.values) <= scatter:
                scatter = len(current_img.values)
           
            for p in np.random.choice(current_img.values, scatter, replace=False):
              plt.scatter(pos_counter+np.random.uniform(-0.1, 0.1), p,
                          s=scatter_size, color=newcolors[np.random.randint(len(newcolors))])
                          
        idx_counter += 1
        pos_counter += 1
      xtick_idx.append((pos_counter-1)-(len(group)-1)/2)
    
      pos_counter += 1
      cidx += 1
    
    plt.xticks(xtick_idx, group_info.keys(),fontsize=font)
    plt.show()

    return fig
    
    
def make_group_plot(path, group_info, color_lst,
                    measurments = 'median', # or 'mean'
                    scatter=50, scatter_size=5,
                    width=0.5, font=12,
                    figsize=[6,4]):
    # load csv file data
    if measurments == 'median':
        stacked_result = pd.read_csv(os.path.join(path, 'median_stacked_master.csv'), index_col=0)
    elif measurments == 'mean':
        stacked_result = pd.read_csv(os.path.join(path, 'mean_stacked_master.csv'), index_col=0)
    else:
        print('check your measurments input for the coming error!')
    pos_counter = 0
    xtick_idx = []
    viridis = cm.get_cmap('hsv', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    fig = plt.figure(figsize=figsize)
    for group in range(len(group_info.keys())):
      current_img = stacked_result.iloc[:,group].dropna()
    
      violin = plt.violinplot(current_img, positions=[pos_counter],
                              showextrema=False, showmeans=False,widths=[width],
                              showmedians=True,quantiles=[0.25,0.75])
      for pc in violin['bodies']:
          pc.set_facecolor(color_lst[pos_counter])
          pc.set_edgecolor('black')
          pc.set_alpha(0.2)
      vp = violin['cmedians']
      vp.set_edgecolor('black')
      vp.set_linewidth(2)
    
      vp = violin['cquantiles']
      vp.set_edgecolor('blue')
      vp.set_linewidth(1)
    
      xtick_idx.append(pos_counter)
    
      # also makes some scatter here
      # check if there is enough points for scattering
      if len(current_img.values) <= scatter:
                scatter = len(current_img.values)
      for p in np.random.choice(current_img.values, scatter, replace=False):
        plt.scatter(pos_counter+np.random.uniform(-0.1, 0.1), p,
                    s=scatter_size, color=newcolors[np.random.randint(len(newcolors))])
      
      pos_counter += 1
    
    plt.xticks(xtick_idx, group_info.keys(), fontsize=font)
    plt.show()

    return fig
    
    
#for % within WT range   
# add threshold for both median and mean
# input [mean, median] for thresholds
# to make this change, additional master files needed! modify in image class!
def get_dim_ratio(path, measurments = 'median', # or 'mean'
                  intensity_threshold_low_bound=0, intensity_threshold_high_bound=5000):
    
    # load csv file data
    if measurments == 'median':
        stacked_results = pd.read_csv(os.path.join(path, 'median_stacked_master.csv'), index_col=0)
    elif measurments == 'mean':
        stacked_results = pd.read_csv(os.path.join(path, 'mean_stacked_master.csv'), index_col=0)
    else:
        print('check your measurments input for the coming error!')
        
    group_precentage = []
    for group in stacked_results.columns:
      low = len(stacked_results[(stacked_results[group] < intensity_threshold_high_bound) & (stacked_results[group] > intensity_threshold_low_bound)])
      #print(round(low))
      #print(round(len(stacked_results[stacked_results[group] > 0])))
      group_precentage.append(100*low/len(stacked_results[stacked_results[group] > 0]))
    
    # load csv file data
    if measurments == 'median':
        sorted_results = pd.read_csv(os.path.join(path, 'median_sorted_master.csv'), index_col=0)
    elif measurments == 'mean':
        sorted_results = pd.read_csv(os.path.join(path, 'mean_sorted_master.csv'), index_col=0)
    else:
        print('check your measurments input for the coming error!')
        
    ratio_dict = {}
    for col in sorted_results.columns:
      current_group = col.split('_')[0]
      low = len(sorted_results[(sorted_results[col] < intensity_threshold_high_bound) & (sorted_results[col] > intensity_threshold_low_bound)])
      low_ratio = 100*low/len(sorted_results[sorted_results[col] > 0])
      print(round(low_ratio,4), col)
      if current_group not in ratio_dict.keys():
        ratio_dict[current_group] = [low_ratio]
      else:
        ratio_dict[current_group].append(low_ratio)
    
    dim_ratio_df = pd.DataFrame(ratio_dict.values()).transpose()
    dim_ratio_df.columns = ratio_dict.keys()
    group_df = pd.DataFrame(group_precentage).transpose()
    group_df.columns = ratio_dict.keys()
    group_df.index = ['Group Ratio']
    
    dim_ratio_df = pd.concat([group_df, dim_ratio_df], axis=0, ignore_index=False,)
    dim_ratio_df.to_csv(os.path.join(path, 'dim_ratio.csv'))
    


def make_mask_overlay(img, masks):

    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)

    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)

    HSV[:,:,2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)

    hues = np.linspace(0, 1, masks.max()+1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        HSV[ipix[0],ipix[1],0] = hues[n]
        HSV[ipix[0],ipix[1],1] = 1.0
        
    return HSV

def make_cmap():
    
    viridis = cm.get_cmap('hsv', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([256/256, 256/256, 256/256, 1])
    newcolors[:1, :] = pink
    newcmp = ListedColormap(newcolors)
    
    return newcmp


def anova_dunn(path):
    stacked_result = pd.read_csv(os.path.join(path, 'stacked_master.csv'), index_col=0)

    # get your significant test results here!

    # apply ANOVA with dunn as posthoc
    anova_dunn_res = sp.posthoc_dunn(stacked_result.values.T, p_adjust = 'holm')
    anova_dunn_res.index = stacked_result.columns
    anova_dunn_res.columns = stacked_result.columns

    # this cell will present sgnifiacnt level in stacked.csv 
    significant_level = sp.sign_table(anova_dunn_res)
    significant_level.index = stacked_result.columns
    significant_level.columns = stacked_result.columns
    
    return anova_dunn_res, significant_level

    
def make_group_label(path, baseline):
  data = pd.read_csv(os.path.join(path, 'sorted_master.csv'), index_col=0)
  res = {baseline:1}
  n=2
  for col in data.columns:
    group = col.split('_')[0]
    if baseline not in col and group not in res.keys():
      res[group] = n
      n+= 1
  return res
  
def orgnize_lme_data(path, group_label):
  data = pd.read_csv(os.path.join(path, 'sorted_master.csv'), index_col=0)
  org_data = pd.DataFrame()
  j=1
  treatment = data.columns[0].split('_')[0]
  for col in data.columns:  
    for key in group_label:
      if key in col:
        n = group_label[key]
    treatment = col.split('_')[0]
    temp_col = data[col].dropna()
    temp_df = pd.DataFrame({'y':temp_col.values,
                            'group':treatment,
                            'treatment':[n]*len(temp_col),
                            'animal_idx':[j]*len(temp_col)})
    org_data = pd.concat([org_data, temp_df],ignore_index=True)
    j+=1
  org_data = org_data.fillna(0)
  return org_data

def print_summary(mdf, group_label):
  other_coef = ''
  for i in range(len(mdf.pvalues)):
    # in our case, we want the treatment result similar to WT
    # which means the distribution of coef. is similar like baseline(WT)
    # in other words, the P should be some thing close to 1
    # On the other hand, it's also possible to set Het to baseline
    # to see if the treamt get significant differnt from the Het
    # however, in such a case, we can't see over dosing intensity
    if 'treatment_idx' in mdf.pvalues.index[i]:
      t_idx = mdf.pvalues.index[i].split('.')[-1][0]
      print('{}: has p-values:{:.4f}'.format(list(group_label.keys())\
                                            [list(group_label.values()).index(int(t_idx))],
                                            mdf.pvalues[i]))
      print('roughly {:.3f}% differnce from {}\n'.format(
          100*mdf.params[i]/abs(mdf.params[i])*(1- mdf.pvalues[i]), 
          list(group_label.keys())[list(group_label.values()).index(min(group_label.values()))]
          )
      )
    else:
      other_coef += str('{} has p-values:{:.4f}\n'.format(mdf.pvalues.index[i],
                                                          mdf.pvalues[i]))
  print(other_coef)

def line(k, x, b):
  return k*x + b

def plot_lme(path, mdf, color_lst, show_plot=True):
  data = pd.read_csv(os.path.join(path, 'stacked_master.csv'), index_col=0)
  fig = plt.figure(figsize=[12,8])

  plt.subplot(221)
  x = np.arange(1,7)
  for i in range(len(data.columns)):
    if i == 0:
      plt.plot(x, line(mdf.fe_params[i+1], x, mdf.fe_params[0]), 
              color=color_lst[i], ls='dashed', label='Baseline', lw=5)
    else:
      plt.plot(x, line(mdf.fe_params[i+1], x, mdf.fe_params[0]), 
              color=color_lst[i], label=data.columns[i])
  plt.legend()

  plt.subplot(212)
  for i in range(len(data.columns)):
    violin_parts = plt.violinplot(data[data.columns[i]].dropna(), positions=[i],
                  showextrema=False)
    for pc in violin_parts['bodies']:
      pc.set_facecolor(color_lst[i])
      pc.set_edgecolor(color_lst[i])

    x = np.arange(-0.4,0.5,0.1)
    plt.plot(x+i, line(mdf.fe_params[i+1], x, mdf.fe_params[0]), color='black')
    plt.plot(x+i, line(mdf.fe_params[1], x, mdf.fe_params[0]), color='red', ls='dashed')

  plt.plot(0, 0, color='black', label='Current slope')
  plt.plot(0, 0, color='red', ls='dashed', label='Baseline slope')

  plt.legend()
  plt.xticks(range(len(data.columns)), data.columns)

  if show_plot:
    plt.show()
  else:
    plt.close()
  return fig
  

if __name__ == '__main__':
    folder_path = sys.argv[1]
    with open('config.yaml') as f:
    	param = yaml.load(f, Loader=yaml.FullLoader)

    batch_image(folder_path, param)
    
    
