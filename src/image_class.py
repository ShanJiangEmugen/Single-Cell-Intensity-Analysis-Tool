##########################################################################
# New version of cellpose masking pipeline
# Goals:
#       1. create a class for entire process
#       2. move all image process in class
#       3. reorgnize code
#       4. optimize processing speed
#
#
##########################################################################
import os
import gc
import sys
import yaml
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


##########################################################################
# useful functions here
def calculate_median(regionmask, intensity):
    return np.median(intensity[regionmask])


def make_dir(path, mask_ch, signal_ch):
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create folder
    folder_name = 'intensity_result_mask{}_signal{}_{}'.format(mask_ch, signal_ch, timestamp)
    os.makedirs(os.path.join(path,folder_name))

    return folder_name
    
    
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
    

##########################################################################
# main class object here
class project:
    def __init__(self, folder, param, cell_size_threshold,group_info):
        
        # get file names from the input folder and sort
        img_fns = [f for f in os.listdir(folder) if not f.startswith('.') and f.endswith('czi')]
        img_fns.sort(key=lambda x:(int(x.split('_')[0]), # index 
                                   int(x.split('_')[-3]), # hour
                                   int(x.split('_')[-2]))) # min
        self.img_fns = img_fns
        print(self.img_fns)
        
        # initialize model
        self.model = models.Cellpose(model_type='cyto', gpu=True)
        self.channels = [[0, 0]]
        
        # keep inputs in record
        self.folder = folder 
        self.cell_size_threshold = cell_size_threshold
        self.group_info = group_info

        self.diameter = param['diameter']
        self.flow_threshold = param['flow_threshold']
        self.cellprob_threshold = param['cellprob_threshold']
        self.min_size = param['min_size']
        
        # create a master output folder here
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = 'analysis_results_{}'.format(timestamp)
        self.out_root = os.path.join(folder,folder_name)
        os.makedirs(self.out_root)
        
        # save parameters first 
        d = {
            'date':timestamp,
            'cell_size_threshold':cell_size_threshold,
            'group_info':group_info,
            'parameters':param
        }
        with open(os.path.join(self.out_root, 'param.yaml'), 'w') as yaml_file:
            yaml.dump(d, yaml_file, default_flow_style=False, sort_keys=False)
        
    def batch_masking(self, cellpose_ch, signal_ch, gamma, save=True):
        # make directory for outputs
        if save:
            self.folder_name = make_dir(self.out_root, cellpose_ch, signal_ch)
            print('Exporting results to: ', self.folder_name)

        # initialize containers for outputs 
        median_master_df = pd.DataFrame()
        mean_master_df = pd.DataFrame()
        col_names = []
        
        
        
        for f in tqdm(range(len(self.img_fns)), desc='Cellpose analyzing progress'):
            # loading image in self defined channels
            img = AICSImage(os.path.join(self.folder, self.img_fns[f]))
            cellpose_masking_ch = img.get_image_data("YX", C=cellpose_ch, S=0, T=0)  # returns 3D ZYX numpy array
            intensity_readout_ch = img.get_image_data("YX", C=signal_ch, S=0, T=0)  # returns 3D ZYX numpy array
            
            # despeckle here
            cellpose_masking_ch = median_filter(cellpose_masking_ch, size=3)  # the default size in ImageJ is 3x3
            intensity_readout_ch = median_filter(intensity_readout_ch, size=3) # this step will takes about 2 seconds for each image(2048x2048)
            
            # main cellpose masking here
            masks, flows, styles, diams = self.model.eval(cellpose_masking_ch, diameter=self.diameter, channels=self.channels,
                                                          flow_threshold=self.flow_threshold,
                                                          cellprob_threshold=self.cellprob_threshold,
                                                          min_size=self.min_size
                                                         )
                                                         
            # here we are done with cellpose masking 
            # we can open up a new function to specify which channel will be masked from mask
            # in this case, each class object will contain 1 mask from 1 channel 
            # if need mask from another channel, a new class object needs to be created!
            
            # however! this part is inside a loop, the mask is a temp varible
            # save it for later use will cause a big time and efficiency concern 
            
            # gamma method for boost contrast rate
            if gamma:
                intensity_readout_ch = intensity_readout_ch/255.0  
                intensity_readout_ch = np.power(intensity_readout_ch,gamma)*255.0
                
            # apply mask to the signal channel 
            props = measure.regionprops_table(masks, intensity_readout_ch, separator='_', 
                                              properties=['label',
                                                          'centroid',
                                                          'area',
                                                          'mean_intensity'],
                                              extra_properties=(calculate_median,))
            props_df = pd.DataFrame(props) # this is df contain all tracked cells with defined properties in pervious lines
            valid_cells = props_df[props_df['area'] >= self.cell_size_threshold] # also filter out some tiny cells 
            
            # export to .csv file in output folder 
            base = os.path.splitext(self.img_fns[f])[0]
            if save:
                valid_cells.to_csv('{}/{}/{}.csv'.format(self.out_root, self.folder_name, base))
            
            # making plots section
            # make all masked cells in same color
            temp_msk = masks.copy()
            temp_msk[temp_msk!=0] = 1
            
            # remove small cells from mask
            small_cells = props_df[props_df['area'] < self.cell_size_threshold]['label'].values
            for cell in small_cells:
              temp_msk[temp_msk==cell]=9999
            
            if save:
                # initialize a big figure for plots
                fig = plt.figure(figsize=[21,10])
                plt.subplot(241)
                plt.imshow(cellpose_masking_ch, cmap='Reds')
                plt.title('Masking Channel')
                
                plt.subplot(242)
                plt.title('Masks over Gery Channel')
                plt.imshow(cellpose_masking_ch, cmap='Reds')
                plt.imshow(temp_msk,cmap='Reds_r')
                # load grey channel just in case
                data_czyx_grey = img.get_image_data("YX", C=1, S=0, T=0)
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
                plt.imshow(intensity_readout_ch, cmap='Reds_r')
                plt.title('Intensity Signal')
                
                plt.subplot(246)
                plt.title('Intensity Signal + Mask')
                plt.imshow(intensity_readout_ch*temp_msk, cmap='Reds_r')
                        
                # add more plot here
                mask_plot = plot.mask_overlay(intensity_readout_ch[:,:], masks)
                outline_plot = plot.outline_view(intensity_readout_ch[:,:],  masks, color=[0,255,0], mode='thick')
                plt.subplot(247)
                plt.imshow(mask_plot, alpha=1)
                plt.title('Intensity Signal + Mask Overlay')
                
                plt.subplot(248)
                plt.imshow(outline_plot.astype(np.uint8))
                plt.title('Intensity Signal + Outline')
        
                plt.tight_layout()
                fig.savefig('{}/{}/{}.png'.format(self.out_root, self.folder_name, base))
                
                plt.clf() # clear fig
                #plt.cla() # clear axes
                # clear figure 
                plt.close(fig)
            
            # save the df for later use /// this part can be optimized! updating columns names df in loop
            col_names.append(base) 
            median_master_df = pd.concat([median_master_df, valid_cells['calculate_median']], axis=1)
            
            # additional df for means 
            mean_master_df = pd.concat([mean_master_df, valid_cells['mean_intensity']], axis=1)
            
            median_master_df.columns = col_names
            mean_master_df.columns = col_names
            # master_df will be saved for later steps: stacked/sorted master csv
            


        return median_master_df, mean_master_df
        
    def orgnize_master_files(self, master_df, mode):
        # make output of means as well
        # however, the master_df was created in last step, make changes there!
        
        # initialize output containers 
        output_master = pd.DataFrame()
        stacked_master_data={}
        col_names = []
        
        # loop through group info to get group number and group names
        for group, indces in self.group_info.items():
            temp_stack = pd.DataFrame()
            
            # just in case the images are not in contiunes order, such as missing #6
            indces_str = list(map(str, indces))
            #indces = np.array(indces)-1
            idx_lst = [f.split('_')[0] for f in master_df.columns]
            true_indces = np.where(np.isin(idx_lst, indces_str))[0]

            current_data = master_df.iloc[:,true_indces]
            output_master = pd.concat([output_master, current_data], axis=1)
            for i in range(len(indces)):
                # make a column name first
                col_names.append('{}_{}'.format(group, indces[i]))
                # then stack all data in 1 column for stacked master
                temp_stack = pd.concat([temp_stack, current_data.iloc[:,i].dropna()], ignore_index=True, axis=0)
            
            stacked_master_data[group] = temp_stack.values.flatten()
        output_master.columns = col_names
        # also sort results by columns names, not groups
        output_master.to_csv('{}/{}/{}_sorted_master.csv'.format(self.out_root, self.folder_name, mode))
        stacked_master = pd.DataFrame(stacked_master_data.values()).transpose()
        stacked_master.columns=stacked_master_data.keys()
        stacked_master.to_csv('{}/{}/{}_stacked_master.csv'.format(self.out_root, self.folder_name, mode))
        
        # outputs of means
        
        
        
        
    # make the comparisons inside the class
    # when the comparisons needed, run this function ONLY
    # this function will do the masking first anyway !!
    
    def both_channel_ratio(self, cellpose_ch, mask_ch1, mask_ch2,
                           low_bound_ch1, up_bound_ch1, low_bound_ch2, up_bound_ch2,
                           group_info=None, measurments='median' # or 'mean'
                           ):
        # make mask frist
        median_master_df, mean_master_df = self.batch_masking(cellpose_ch=cellpose_ch,  # channel for cellpose masking
                                        signal_ch=mask_ch1,  # channel for getting sinal intensity from mask
                                        gamma=1)

        # filter out cells not in defined range
        median_master_df = median_master_df[(median_master_df>low_bound_ch1)&(median_master_df<up_bound_ch1)]
        mean_master_df = mean_master_df[(mean_master_df>low_bound_ch1)&(mean_master_df<up_bound_ch1)]
        
        # save master files here
        self.orgnize_master_files(median_master_df, 'median')
        self.orgnize_master_files(mean_master_df, 'mean')

        median_master_df2, mean_master_df2 = self.batch_masking(cellpose_ch=cellpose_ch,  # channel for cellpose masking
                                            signal_ch=mask_ch2,  # channel for getting sinal intensity from mask
                                            gamma=1)
                                            
        # filter out cells not in defined range
        median_master_df2 = median_master_df2[(median_master_df2>low_bound_ch2)&(median_master_df2<up_bound_ch2)]
        mean_master_df2 = mean_master_df2[(mean_master_df2>low_bound_ch2)&(mean_master_df2<up_bound_ch2)]   
        
        # save master files here
        self.orgnize_master_files(median_master_df2, 'median')
        self.orgnize_master_files(mean_master_df2, 'mean')
        
        # prepare the output folder
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = 'comparison_result_ch1_{}to{}_ch2_{}to{}_{}'.format(low_bound_ch1, up_bound_ch1, 
                                                                          low_bound_ch2, up_bound_ch2,
                                                                          timestamp)
        save_path = os.path.join(self.out_root,folder_name)
        os.makedirs(save_path)  
    
        # initilize output
        compare = pd.DataFrame(index=['ch{} %'.format(mask_ch1), 
                                      'ch{} %'.format(mask_ch2), 
                                      'ch{} in ch{} %'.format(mask_ch1, mask_ch2)])
                                    
        temp_res = []
        for col in median_master_df:
            
            current_file = col
            # also consider mean as input as well
            if measurments == 'median':
                df1 = median_master_df[col]
                df2 = median_master_df2[col]
            elif measurments == 'mean':
                df1 = mean_master_df[col]
                df2 = mean_master_df2[col]
            else: 
                print('check your measurments input!')
                
            df = pd.concat([df1, df2], axis=1, ignore_index=False)
            df.columns = ['ch1', 'ch2']
            df = df.dropna(how='all')
            # check df before running
            if not len(df) > 0:
                print('there is no cells in this image: {}'.format(col))
              
            else:
                  # calculate percentage here 
                  # 1. % in first channel in [low, up]
                  # 2. % in second channel in [low, up]
                  # 3. % in second channel based on first channel in [low, up]
                
                  # 1. apply thereshold for ch1
                  temp1 = df[(df['ch1']>=low_bound_ch1)&(df['ch1']<=up_bound_ch1)]
                  p1 = len(temp1)/len(df)
                    
                  # 2. apply thereshold for ch2
                  temp2 = df[(df['ch2']>=low_bound_ch2)&(df['ch2']<=up_bound_ch2)]
                  p2 = len(temp2)/len(df)
                
                  # 3. in range ch2 in ch1
                  temp3 = temp1[(temp1['ch2']>=low_bound_ch2)&(temp1['ch2']<=up_bound_ch2)]
                  if len(temp1) > 0 :
                        p3 = len(temp3)/len(temp1)
                  else:
                        print('Warning: no cells in ch1')
                        p3 = np.nan
                      
                  compare[col] = [p1, p2, p3]
              
            compare.to_csv(os.path.join(save_path, 'raw_comparison.csv'))
            
        # orgnize output based on group info here:
        # sorted master: sort by groups 
        # stacked master_df: stack all data from same group into 1 column 
        if group_info != None:
            sorted_master = pd.DataFrame()
            stacked_master = {}
            stacked_master_df = pd.DataFrame()
            col_names = []
            
            for group, indces in group_info.items():
                temp_stack = pd.DataFrame()
                # get each group data 
                indces = np.array(indces)-1
                current_data = compare.iloc[:,indces]
    
                # save a group as a column
                current_data_sort = current_data.T
                current_data_sort['group'] = group
                current_data_sort = current_data_sort.T
                sorted_master = pd.concat([sorted_master, current_data_sort], axis=1)
                for i in range(len(indces)):
                    # make a column name first
                    col_names.append('{}_{}'.format(group, indces[i]+1))
                    # then stack all data in 1 column for stacked master
                    temp_stack = pd.concat([temp_stack, current_data.iloc[:,i].dropna()], ignore_index=True, axis=0)
                stacked_master[group] = temp_stack.values.flatten()
            stacked_master_df = pd.DataFrame(stacked_master.values()).transpose()
            stacked_master_df.columns=stacked_master.keys()
            sorted_master = sorted_master.T.set_index(['group',sorted_master.T.index,]).T
                
            stacked_master_df.to_csv(os.path.join(save_path, 'stacked_comparison.csv'))
            sorted_master.to_csv(os.path.join(save_path, 'sorted_comparison.csv'))
        
            print('\nDone!')
            
            
            
##########################################################################
# added on Nov 19, 2025
# new feature added: count cells in all channels for further analysis
# return a new csv file in previous structure
# only show counts in each channel for all input images
# 


    def count_cells_per_channel(self, cellpose_ch_list, save=True):
        """
        对每张图片的每个指定 channel 单独跑一次 cellpose，
        按 cell_size_threshold 过滤后，统计每个 channel 的细胞数量。

        结果：
        - raw_counts: 列为每张图片（文件名 base），行为 chX
        - sorted_counts: 列为 group_index（如 WT_1, KO_2），行为 chX
        """
        # 先存原始统计（列 = 图片，行 = channel）
        raw_counts = pd.DataFrame()

        for f, fn in enumerate(tqdm(self.img_fns, desc='Counting cells per channel')):
            img_path = os.path.join(self.folder, fn)
            img = AICSImage(img_path)
            base = os.path.splitext(fn)[0]

            # 每个 channel 的 cell 数量
            counts_one_image = []

            for ch in cellpose_ch_list:
                # 读取该 channel 的图像
                ch_img = img.get_image_data("YX", C=ch, S=0, T=0)

                # despeckle
                ch_img = median_filter(ch_img, size=3)

                # 跑 cellpose
                masks, flows, styles, diams = self.model.eval(
                    ch_img,
                    diameter=self.diameter,
                    channels=self.channels,
                    flow_threshold=self.flow_threshold,
                    cellprob_threshold=self.cellprob_threshold,
                    min_size=self.min_size
                )

                # 统计每个 label 的面积，并用 cell_size_threshold 过滤
                props = measure.regionprops_table(
                    masks,
                    properties=['label', 'area']
                )
                props_df = pd.DataFrame(props)
                valid_cells = props_df[props_df['area'] >= self.cell_size_threshold]
                counts_one_image.append(len(valid_cells))

            # 将当前图片的结果作为一列
            raw_counts[base] = counts_one_image

        # 设置行索引为 chX
        raw_counts.index = [f'ch{ch}' for ch in cellpose_ch_list]

        # 如果有 group_info，就按 group_info 生成 sorted 格式
        if self.group_info is not None:
            sorted_counts = pd.DataFrame(index=raw_counts.index)
            col_names = []

            # 文件名前缀（通常是 index，例如 "1_xxx.czi" 的 "1"）
            idx_lst = [c.split('_')[0] for c in raw_counts.columns]

            for group, indices in self.group_info.items():
                indices_str = list(map(str, indices))
                # 找到属于该 group 的列
                mask = np.isin(idx_lst, indices_str)
                current_data = raw_counts.loc[:, mask]

                # 按 group 拼接
                sorted_counts = pd.concat([sorted_counts, current_data], axis=1)

                # 生成列名：group_index（例如 WT_1, KO_3）
                for col in raw_counts.columns[mask]:
                    img_idx = col.split('_')[0]  # 文件名前缀当作 index
                    col_names.append(f'{group}_{img_idx}')

            sorted_counts.columns = col_names
        else:
            sorted_counts = raw_counts.copy()

        if save:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_name = f'cell_counts_per_channel_{timestamp}'
            save_path = os.path.join(self.out_root, folder_name)
            os.makedirs(save_path, exist_ok=True)

            # 原始按图片的统计
            raw_counts.to_csv(os.path.join(save_path, 'cell_counts_raw.csv'))
            # 按 group_info 排序后的统计（你要的格式）
            sorted_counts.to_csv(os.path.join(save_path, 'cell_counts_sorted.csv'))

            print(f'\nCell count results saved to: {save_path}')

        return raw_counts, sorted_counts

        