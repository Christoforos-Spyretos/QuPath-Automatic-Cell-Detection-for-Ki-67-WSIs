# %% IMPORTS
import os
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import openslide
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter

# %% PATH TO IMGS
img1_pos = '/local/data1/chrsp39/QuPath_Portable/results/C52890___7316-1460___Ki-67_A2_PosDMap.tif'
img1_neg = '/local/data1/chrsp39/QuPath_Portable/results/C52890___7316-1460___Ki-67_A2_NegDMap.tif'
img1 = '/local/data2/chrsp39/CBTN_v2/new_KI67/WSI/C52890___7316-1460___Ki-67_A2.svs'

img2_pos = '/local/data1/chrsp39/QuPath_Portable/results/C47724___7316-490___Ki-67_PosDMap.tif'
img2_neg = '/local/data1/chrsp39/QuPath_Portable/results/C47724___7316-490___Ki-67_NegDMap.tif'
img2 = '/local/data2/chrsp39/CBTN_v2/new_KI67/WSI/C47724___7316-490___Ki-67.svs'

img3_pos = '/local/data1/chrsp39/QuPath_Portable/results/C48831___7316-2985___Ki-67_A2_PosDMap.tif'
img3_neg = '/local/data1/chrsp39/QuPath_Portable/results/C48831___7316-2985___Ki-67_A2_NegDMap.tif'
img3 = '/local/data2/chrsp39/CBTN_v2/new_KI67/WSI/C48831___7316-2985___Ki-67_A2.svs'

img4_pos = '/local/data1/chrsp39/QuPath_Portable/results/C102459___7316-475___Ki-67_PosDMap.tif'
img4_neg = '/local/data1/chrsp39/QuPath_Portable/results/C102459___7316-475___Ki-67_NegDMap.tif'
img4 = '/local/data2/chrsp39/CBTN_v2/new_KI67/WSI/C102459___7316-475___Ki-67.svs'

img5_pos = '/local/data1/chrsp39/QuPath_Portable/results/C54735___7316-288___Ki-67_C_PosDMap.tif'
img5_neg = '/local/data1/chrsp39/QuPath_Portable/results/C54735___7316-288___Ki-67_C_NegDMap.tif'
img5 = '/local/data2/chrsp39/CBTN_v2/new_KI67/WSI/C54735___7316-288___Ki-67_C.svs'

img6_pos = '/local/data1/chrsp39/QuPath_Portable/results/C18573___7316-102___Ki-67_(B)_PosDMap.tif'
img6_neg = '/local/data1/chrsp39/QuPath_Portable/results/C18573___7316-102___Ki-67_(B)_NegDMap.tif'
img6 = '/local/data2/chrsp39/CBTN_v2/new_KI67/WSI/C18573___7316-102___Ki-67_(B).svs'

img7_pos = '/local/data1/chrsp39/QuPath_Portable/results/C99753___7316-466___Ki-67_PosDMap.tif'
img7_neg = '/local/data1/chrsp39/QuPath_Portable/results/C99753___7316-466___Ki-67_NegDMap.tif'
img7 = '/local/data2/chrsp39/CBTN_v2/new_KI67/WSI/C99753___7316-466___Ki-67.svs'

img8_pos = '/local/data1/chrsp39/QuPath_Portable/results/C18819___7316-41___Ki-67_PosDMap.tif'
img8_neg = '/local/data1/chrsp39/QuPath_Portable/results/C18819___7316-41___Ki-67_NegDMap.tif'
img8 = '/local/data2/chrsp39/CBTN_v2/new_KI67/WSI/C18819___7316-41___Ki-67.svs'

pos_density_maps = [img1_pos, img2_pos, img3_pos, img4_pos, img5_pos, img6_pos, img7_pos, img8_pos]
neg_density_maps = [img1_neg, img2_neg, img3_neg, img4_neg, img5_neg, img6_neg, img7_neg, img8_neg]
img = [img1, img2, img3, img4, img5, img6, img7, img8]
slide_ids = ['C52890___7316-1460___Ki-67_A2', 'C47724___7316-490___Ki-67', 'C48831___7316-2985___Ki-67_A2',
             'C102459___7316-475___Ki-67', 'C54735___7316-288___Ki-67_C', 'C18573___7316-102___Ki-67_(B)',
             'C99753___7316-466___Ki-67', 'C18819___7316-41___Ki-67']

# %% OPEN WSI
slide = openslide.OpenSlide(img5)
wsi = slide.read_region((0, 0), 0, slide.level_dimensions[0])

#shape
print(wsi.size)

im_neg = Image.open(img5_neg)  
width, height = im_neg.size

# resize WSI
wsi_resized = wsi.resize((width, height), Image.Resampling.LANCZOS)

# %% PLOT WSI
fig, ax = plt.subplots()
plt.imshow(wsi_resized)
plt.xticks([])
plt.yticks([])
# plt.title('WSI')
plt.show()

# %% NEGATIVE CELL DENSITY MAP 
im_neg = Image.open(img8_neg)  

# plot the orginal negative cell density map
fig, ax = plt.subplots()
plt.imshow(im_neg)
plt.xticks([])
plt.yticks([])
plt.title('Original Grayscale Negative Cell Density Map')
plt.show()

width, height = im_neg.size

if width < height:
    orient = 'vertical'
else:
    orient = 'horizontal'

imarray_neg = np.array(im_neg)

cm_lim_neg = np.max(imarray_neg)
new_imarray_neg = imarray_neg / cm_lim_neg
new_imarray_neg[new_imarray_neg>1] = 1 

cmapQ_neg = plt.get_cmap('jet')
A_neg = cmapQ_neg(new_imarray_neg)

# convert dark blue background to white
A_neg[new_imarray_neg == 0] = [1, 1, 1, 1]

# plot
fig, ax = plt.subplots()
im = ax.imshow(A_neg)
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title('Negative Cell Density Map')
divider = make_axes_locatable(ax)
if orient == 'vertical':
    cax = divider.append_axes("right", size="5%", pad=0.20)
else:
    cax = divider.append_axes("bottom", size="5%", pad=0.20)
norm = matplotlib.colors.Normalize(vmin=0, vmax=2688.02781739274)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
plt.show()

# %% POSITIVE CELL DENSITY MAP
im_pos = Image.open(img8_pos)

# plot the orginal negative cell density map
fig, ax = plt.subplots()
plt.imshow(im_pos)
plt.xticks([])
plt.yticks([])
plt.title('Original Grayscale Positive Cell Density Map')
plt.show()

width, height = im_pos.size

if width < height:
    orient = 'vertical'
else:
    orient = 'horizontal'

imarray_pos = np.array(im_pos)
cm_lim_pos = np.max(imarray_pos)
new_imarray_pos = imarray_pos / cm_lim_pos
new_imarray_pos[new_imarray_pos>1] = 1 

cmapQ_pos = plt.get_cmap('jet')
A_pos = cmapQ_pos(new_imarray_pos)

# convert dark blue background to white
A_pos[new_imarray_pos == 0] = [1, 1, 1, 1]

fig, ax = plt.subplots()
im = ax.imshow(A_pos)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Positive Cell Density Map (From Juan)')
divider = make_axes_locatable(ax)
if orient == 'vertical':
    cax = divider.append_axes("right", size="5%", pad=0.20)
else:
    cax = divider.append_axes("bottom", size="5%", pad=0.20)
norm = matplotlib.colors.Normalize(vmin=0, vmax=cm_lim_pos)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
plt.show()

# all non-white pixels in A_neg are set to dark blue
A_neg_blue_mask = A_neg
A_neg_blue_mask[np.any(A_neg_blue_mask[:, :, :3] != [1, 1, 1], axis=2)] = [0, 0, 1, 1]  # Dark blue

fig, ax = plt.subplots()
plt.imshow(A_neg_blue_mask)
plt.title('Mask of the Negative Cell Density Maps')
plt.xticks([])
plt.yticks([])
plt.show()

new_A_pos = A_pos
white_mask = np.all(new_A_pos[:, :, :3] == [1, 1, 1], axis=2)
new_A_pos[white_mask] = A_neg_blue_mask[white_mask]

fig, ax = plt.subplots()
im = ax.imshow(new_A_pos)
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title('Positive Cell Density Map')
divider = make_axes_locatable(ax)
if orient == 'vertical':
    cax = divider.append_axes("right", size="5%", pad=0.20)
else:
    cax = divider.append_axes("bottom", size="5%", pad=0.20)
norm = matplotlib.colors.Normalize(vmin=0, vmax=2618.66443984104)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
plt.show()

# %% Ki-67 POSITIVE NEGATIVE RATIO MAP
neg_im = Image.open(img8_neg)
width, height = neg_im.size

if width < height:
    orient = 'vertical'
else:
    orient = 'horizontal'

    
neg_imarray = np.array(neg_im)

pos_im = Image.open(img8_pos)
pos_imarray = np.array(pos_im)

for k in range(np.shape(neg_imarray)[0]):
    for l in range(np.shape(neg_imarray)[1]):
        denominator = neg_imarray[k, l] + pos_imarray[k, l]
        # if denominator is 0, assign a default value
        if denominator == 0:
            neg_imarray[k, l] = 0.0001 
        else:
            neg_imarray[k, l] = pos_imarray[k, l] / denominator
        if neg_imarray[k, l] < 0.0001:
            neg_imarray[k, l] = 0.0001
        # elif neg_imarray[k, l] > 1:
        #     neg_imarray[k, l] = 1
        elif math.isnan(neg_imarray[k, l]):
            neg_imarray[k, l] = 0.0001
max_rat = np.max(neg_imarray[neg_imarray<1])

# only take into account foreground pixels
relevant_values = neg_imarray[neg_imarray > 0] 
n_relval = len(relevant_values)
adjusted_n_relval = round(n_relval*0.95)
sorted_relval = sorted(relevant_values)
min_cm_lim = 0.03
cm_lim = max(max(sorted_relval[:adjusted_n_relval]), min_cm_lim)
cm_lim = max(sorted_relval[:adjusted_n_relval])
neg_imarray = neg_imarray/cm_lim
neg_imarray[neg_imarray>1] = 1 

cmapQ = plt.get_cmap('jet')
A_pos_neg_ratio = cmapQ(neg_imarray)

# convert dark blue background to white
dark_blue = [0, 0, 0.5, 1] # RGBA values for dark blue
white = [1, 1, 1, 1] # RGBA values for white
A_pos_neg_ratio[
    (A_pos_neg_ratio[:, :, 0] == dark_blue[0]) & 
    (A_pos_neg_ratio[:, :, 1] == dark_blue[1]) & 
    (A_pos_neg_ratio[:, :, 2] == dark_blue[2]) & 
    (A_pos_neg_ratio[:, :, 3] == dark_blue[3])
    ] = white

new_A_pos_neg_ratio = A_pos_neg_ratio
white_mask = np.all(new_A_pos_neg_ratio[:, :, :3] == [1, 1, 1], axis=2)
new_A_pos_neg_ratio[white_mask] = A_neg_blue_mask[white_mask]

fig, ax = plt.subplots()
im = ax.imshow(new_A_pos_neg_ratio)
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title('Ki-67 LI Cell Density Map')
divider = make_axes_locatable(ax)
if orient == 'vertical':
    cax = divider.append_axes("right", size="5%", pad=0.20)
else:
    cax = divider.append_axes("bottom", size="5%", pad=0.20)
norm = matplotlib.colors.Normalize(vmin=0, vmax=49.35)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
plt.show()

# %% FOR LOOP 
results = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/QuPath_Ki-67_summary_analysis.csv')



for imgs in zip(pos_density_maps, neg_density_maps, img, slide_ids):

    pos_density_map = imgs[0]
    neg_density_map = imgs[1]
    wsi = imgs[2]
    slide_id = imgs[3]

    neg_density = round(results[results['slide_id'] == slide_id]['Neg_Density'].values[0], 2)
    pos_density = round(results[results['slide_id'] == slide_id]['Pos_Density'].values[0], 2)
    ki67_li = round(results[results['slide_id'] == slide_id]['Pos_Percentage'].values[0], 2)

    # Open WSI
    slide = openslide.OpenSlide(wsi)
    wsi = slide.read_region((0, 0), 0, slide.level_dimensions[0])
    
    # Resize WSI
    im_neg = Image.open(neg_density_map)  
    width, height = im_neg.size
    wsi_resized = wsi.resize((width, height), Image.Resampling.LANCZOS)

    # Plot WSI
    fig, ax = plt.subplots()
    plt.imshow(wsi_resized)
    plt.xticks([])
    plt.yticks([])
    plt.title('WSI (Resized to the size of the density maps)')
    plt.savefig(f'./{slide_id}_WSI.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Negative Cell Density Map
    im_neg = Image.open(neg_density_map)
    # Plot the original negative cell density map
    fig, ax = plt.subplots()
    plt.imshow(im_neg)
    plt.xticks([])
    plt.yticks([])
    plt.title('Original Grayscale Negative Cell Density Map')
    plt.savefig(f'./{slide_id}_OriginalNegDMap.png', dpi=300, bbox_inches='tight')
    plt.show()


    width, height = im_neg.size
    if width < height:
        orient = 'vertical'
    else:
        orient = 'horizontal'

    imarray_neg = np.array(im_neg)
    cm_lim_neg = np.max(imarray_neg)

    new_imarray_neg = imarray_neg / cm_lim_neg
    new_imarray_neg[new_imarray_neg>1] = 1

    cmapQ_neg = plt.get_cmap('jet')
    A_neg = cmapQ_neg(new_imarray_neg)
    # convert dark blue background to white
    A_neg[new_imarray_neg == 0] = [1, 1, 1, 1]

    fig, ax = plt.subplots()
    im = ax.imshow(A_neg)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title('Negative Cell Density Map')
    divider = make_axes_locatable(ax)
    if orient == 'vertical':
        cax = divider.append_axes("right", size="5%", pad=0.20)
    else:
        cax = divider.append_axes("bottom", size="5%", pad=0.20)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=neg_density)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
    plt.savefig(f'./{slide_id}_NegDMap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # positive Cell Density Map
    im_pos = Image.open(pos_density_map)

    # plot the original positive cell density map
    fig, ax = plt.subplots()
    plt.imshow(im_pos)
    plt.xticks([])
    plt.yticks([])
    plt.title('Original Grayscale Positive Cell Density Map')
    plt.savefig(f'./{slide_id}_OriginalPosDMap.png', dpi=300, bbox_inches='tight')
    plt.show()

    width, height = im_pos.size
    if width < height:
        orient = 'vertical'
    else:
        orient = 'horizontal'

    imarray_pos = np.array(im_pos)
    cm_lim_pos = np.max(imarray_pos)

    new_imarray_pos = imarray_pos / cm_lim_pos
    new_imarray_pos[new_imarray_pos>1] = 1
    
    cmapQ_pos = plt.get_cmap('jet')
    A_pos = cmapQ_pos(new_imarray_pos)
    # convert dark blue background to white
    A_pos[new_imarray_pos == 0] = [1, 1, 1, 1]

    # fig, ax = plt.subplots()
    # im = ax.imshow(A_pos)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_title('Positive Cell Density Map (From Juan)')
    # divider = make_axes_locatable(ax)
    # if orient == 'vertical':
    #     cax = divider.append_axes("right", size="5%", pad=0.20)
    # else:
    #     cax = divider.append_axes("bottom", size="5%", pad=0.20)
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=cm_lim_pos)
    # cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
    # plt.savefig(f'./{slide_id}_OldPosDMap.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # all non-white pixels in A_neg are set to dark blue
    A_neg_blue_mask = A_neg
    A_neg_blue_mask[np.any(A_neg_blue_mask[:, :, :3] != [1, 1, 1], axis=2)] = [0, 0, 1, 1]  # Dark blue

    # fig, ax = plt.subplots()
    # plt.imshow(A_neg_blue_mask)
    # plt.title('Mask of the Negative Cell Density Maps')
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig(f'./{slide_id}_MaskNegDMap.png', dpi=300, bbox_inches='tight')
    # plt.show()

    new_A_pos = A_pos
    white_mask = np.all(new_A_pos[:, :, :3] == [1, 1, 1], axis=2)
    new_A_pos[white_mask] = A_neg_blue_mask[white_mask]

    fig, ax = plt.subplots()
    im = ax.imshow(new_A_pos)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title('Positive Cell Density Map')
    divider = make_axes_locatable(ax)
    if orient == 'vertical':
        cax = divider.append_axes("right", size="5%", pad=0.20)
    else:
        cax = divider.append_axes("bottom", size="5%", pad=0.20)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=pos_density)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
    plt.savefig(f'./{slide_id}_NewPosDMap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Ki-67 Positive Negative Ratio Map
    neg_im = Image.open(neg_density_map)
    width, height = neg_im.size
    if width < height:
        orient = 'vertical'
    else:
        orient = 'horizontal'

    neg_imarray = np.array(neg_im)
    pos_im = Image.open(pos_density_map)
    pos_imarray = np.array(pos_im)
    for k in range(np.shape(neg_imarray)[0]):
        for l in range(np.shape(neg_imarray)[1]):
            denominator = neg_imarray[k, l] + pos_imarray[k, l]
            # if denominator is 0, assign a default value
            if denominator == 0:
                neg_imarray[k, l] = 0.0001 
            else:
                neg_imarray[k, l] = pos_imarray[k, l] / denominator
            if neg_imarray[k, l] < 0.0001:
                neg_imarray[k, l] = 0.0001
            elif neg_imarray[k, l] > 1:
                neg_imarray[k, l] = 1
            elif math.isnan(neg_imarray[k, l]):
                neg_imarray[k, l] = 0.0001
    # max_rat = np.max(neg_imarray[neg_imarray<1])

    # only take into account foreground pixels
    relevant_values = neg_imarray[neg_imarray > 0]
    n_relval = len(relevant_values)
    adjusted_n_relval = round(n_relval*0.95)
    sorted_relval = sorted(relevant_values)
    min_cm_lim = 0.03
    cm_lim = max(max(sorted_relval[:adjusted_n_relval]), min_cm_lim)
    # cm_lim = (max(sorted_relval[:adjusted_n_relval]))
    neg_imarray = neg_imarray/cm_lim
    neg_imarray[neg_imarray>1] = 1

    cmapQ = plt.get_cmap('jet')
    A_pos_neg_ratio = cmapQ(neg_imarray)

    # convert dark blue background to white
    dark_blue = [0, 0, 0.5, 1] # RGBA values for dark blue
    white = [1, 1, 1, 1] # RGBA values for white
    A_pos_neg_ratio[
        (A_pos_neg_ratio[:, :, 0] == dark_blue[0]) & 
        (A_pos_neg_ratio[:, :, 1] == dark_blue[1]) & 
        (A_pos_neg_ratio[:, :, 2] == dark_blue[2]) & 
        (A_pos_neg_ratio[:, :, 3] == dark_blue[3])
        ] = white

    # old positive negative ratio map
    # fig, ax = plt.subplots()
    # im = ax.imshow(A_pos_neg_ratio)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_title('Positive Negative Ratio Cell Density Map (Juan)')
    # divider = make_axes_locatable(ax)
    # if orient == 'vertical':
    #     cax = divider.append_axes("right", size="5%", pad=0.20)
    # else:
    #     cax = divider.append_axes("bottom", size="5%", pad=0.20)
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=cm_lim)
    # cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
    # plt.savefig(f'./{slide_id}_OldPosNegRatioMap.png', dpi=300, bbox_inches='tight')
    # plt.show()

    new_A_pos_neg_ratio = A_pos_neg_ratio
    white_mask = np.all(new_A_pos_neg_ratio[:, :, :3] == [1, 1, 1], axis=2)
    new_A_pos_neg_ratio[white_mask] = A_neg_blue_mask[white_mask]

    # new positive negative ratio map
    fig, ax = plt.subplots()
    im = ax.imshow(new_A_pos_neg_ratio)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title('Positive Negative Ratio Cell Density Map')
    divider = make_axes_locatable(ax)
    if orient == 'vertical':
        cax = divider.append_axes("right", size="5%", pad=0.20)
    else:
        cax = divider.append_axes("bottom", size="5%", pad=0.20)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=ki67_li)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
    plt.savefig(f'./{slide_id}_NewPosNegRatioMap.png', dpi=300, bbox_inches='tight')
    plt.show()
    

# %% DENSITY MAPS
import os
import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import openslide
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

path_to_imgs = '/local/data1/chrsp39/QuPath_Portable/results'
save_path = '/local/data1/chrsp39/QuPath_Portable/Density_Maps' 
results = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/QuPath_Ki-67_summary_analysis.csv')

if not os.path.exists(save_path):
    os.makedirs(save_path)

imgs = os.listdir(path_to_imgs)

neg_density_maps = []
for img in imgs:
    img = img.split('.')[0]
    if 'NegDMap' in img:
        neg_density_maps.append(img)

pos_density_maps = []
for img in imgs:
    img = img.split('.')[0]
    if 'PosDMap' in img:
        pos_density_maps.append(img)

for neg_density_map in neg_density_maps:
    slide_id = neg_density_map.split('_NegDMap')[0]
    neg_density = round(results[results['slide_id'] == slide_id]['Neg_Density'].values[0], 2)
    pos_density = round(results[results['slide_id'] == slide_id]['Pos_Density'].values[0], 2)
    ki67_li = round(results[results['slide_id'] == slide_id]['Pos_Percentage'].values[0], 2)


    neg_density_map = os.path.join(path_to_imgs, neg_density_map + '.tif')
    im_neg = Image.open(neg_density_map)  

    width, height = im_neg.size

    if width < height:
        orient = 'vertical'
    else:
        orient = 'horizontal'

    imarray_neg = np.array(im_neg)

    cm_lim_neg = np.max(imarray_neg)
    new_imarray_neg = imarray_neg / cm_lim_neg
    new_imarray_neg[new_imarray_neg>1] = 1 

    cmapQ_neg = plt.get_cmap('jet')
    A_neg = cmapQ_neg(new_imarray_neg)

    # convert dark blue background to white
    A_neg[new_imarray_neg == 0] = [1, 1, 1, 1]

    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(A_neg)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Negative Cell Density Map')
    divider = make_axes_locatable(ax)
    if orient == 'vertical':
        cax = divider.append_axes("right", size="5%", pad=0.20)
    else:
        cax = divider.append_axes("bottom", size="5%", pad=0.20)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=neg_density)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
    plt.savefig(f'{save_path}/{slide_id}_NegDMap.png', dpi=300, bbox_inches='tight')

    # mask of the negative cell density map
    A_neg_blue_mask = A_neg
    A_neg_blue_mask[np.any(A_neg_blue_mask[:, :, :3] != [1, 1, 1], axis=2)] = [0, 0, 1, 1]  # dark blue

    # positive cell density map
    # get the corresponding positive cell density map with the same slide_id
    pos_density_map = slide_id + '_PosDMap'

    if pos_density_map in pos_density_maps:
        pos_density_map = os.path.join(path_to_imgs, pos_density_map + '.tif')
        im_pos = Image.open(pos_density_map)

        width, height = im_pos.size

        if width < height:
            orient = 'vertical'
        else:
            orient = 'horizontal'

        imarray_pos = np.array(im_pos)
        cm_lim_pos = np.max(imarray_pos)
        new_imarray_pos = imarray_pos / cm_lim_pos
        new_imarray_pos[new_imarray_pos>1] = 1 

        cmapQ_pos = plt.get_cmap('jet')
        A_pos = cmapQ_pos(new_imarray_pos)

        # convert dark blue background to white
        A_pos[new_imarray_pos == 0] = [1, 1, 1, 1]

        new_A_pos = A_pos
        white_mask = np.all(new_A_pos[:, :, :3] == [1, 1, 1], axis=2)
        new_A_pos[white_mask] = A_neg_blue_mask[white_mask]

        fig, ax = plt.subplots()
        im = ax.imshow(new_A_pos)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Positive Cell Density Map')
        divider = make_axes_locatable(ax)
        if orient == 'vertical':
            cax = divider.append_axes("right", size="5%", pad=0.20)
        else:
            cax = divider.append_axes("bottom", size="5%", pad=0.20)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=pos_density)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
        plt.savefig(f'{save_path}/{slide_id}_PosDensMap.png', dpi=300, bbox_inches='tight')
    
    # Ki-67 LI Map
    for k in range(np.shape(imarray_neg)[0]):
        for l in range(np.shape(imarray_neg)[1]):
            denominator = imarray_neg[k, l] + imarray_pos[k, l]
            # if denominator is 0, assign a default value
            if denominator == 0:
                imarray_neg[k, l] = 0.0001 
            else:
                imarray_neg[k, l] = imarray_pos[k, l] / denominator
            if imarray_neg[k, l] < 0.0001:
                imarray_neg[k, l] = 0.0001
            elif imarray_neg[k, l] > 1:
                imarray_neg[k, l] = 1
            elif math.isnan(imarray_neg[k, l]):
                imarray_neg[k, l] = 0.0001
    # max_rat = np.max(neg_imarray[neg_imarray<1])

    # only take into account foreground pixels
    relevant_values = imarray_neg[imarray_neg > 0]
    n_relval = len(relevant_values)
    adjusted_n_relval = round(n_relval*0.95)
    sorted_relval = sorted(relevant_values)
    min_cm_lim = 0.03
    cm_lim = max(max(sorted_relval[:adjusted_n_relval]), min_cm_lim)
    # cm_lim = (max(sorted_relval[:adjusted_n_relval]))
    imarray_neg = imarray_neg/cm_lim
    imarray_neg[imarray_neg>1] = 1

    cmapQ = plt.get_cmap('jet')
    A_pos_neg_ratio = cmapQ(imarray_neg)

    # convert dark blue background to white
    dark_blue = [0, 0, 0.5, 1] # RGBA values for dark blue
    white = [1, 1, 1, 1] # RGBA values for white
    A_pos_neg_ratio[
        (A_pos_neg_ratio[:, :, 0] == dark_blue[0]) & 
        (A_pos_neg_ratio[:, :, 1] == dark_blue[1]) & 
        (A_pos_neg_ratio[:, :, 2] == dark_blue[2]) & 
        (A_pos_neg_ratio[:, :, 3] == dark_blue[3])
        ] = white

    new_A_pos_neg_ratio = A_pos_neg_ratio
    white_mask = np.all(new_A_pos_neg_ratio[:, :, :3] == [1, 1, 1], axis=2)
    new_A_pos_neg_ratio[white_mask] = A_neg_blue_mask[white_mask]

    # Ki-67 LI map
    fig, ax = plt.subplots()
    im = ax.imshow(new_A_pos_neg_ratio)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Ki-67 LI Map')
    divider = make_axes_locatable(ax)
    if orient == 'vertical':
        cax = divider.append_axes("right", size="5%", pad=0.20)
    else:
        cax = divider.append_axes("bottom", size="5%", pad=0.20)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=ki67_li)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation=orient, shrink=1.0)
    plt.savefig(f'{save_path}/{slide_id}_Ki67_LI_map.png', dpi=300, bbox_inches='tight')
# %%