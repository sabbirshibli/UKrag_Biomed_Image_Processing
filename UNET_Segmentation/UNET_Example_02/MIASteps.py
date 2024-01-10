#Libraries to use
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
#Change slash \ for /
def change_slash(path):
    correct_path=''
    for char in path:
        if char != "\\":
            correct_path=correct_path+char
        else:
            correct_path=correct_path+"/"
    return correct_path
#Visualize planes: sagital, axial, and coronal
def Visualize(vol):
 view_1v2=vol[96,:,:]#Axial View
 view_0v2=vol[:,128,:]#Coronal View
 xfm1=ndi.rotate(view_0v2,angle=-90)
 view_0v1=vol[:,:,128]#Sagital view
 xfm2=ndi.rotate(view_0v1,angle=-90)
 fig,axes=plt.subplots(nrows=1,ncols=3)
 axes[0].imshow(view_1v2,cmap='gray',)
 axes[0].set_title('Sagital View')
 axes[1].imshow(xfm1,cmap='gray')
 axes[1].set_title('Axial View')
 axes[2].imshow(xfm2,cmap='gray')
 axes[2].set_title('Coronal View')
 for ax in axes:
    ax.axis('off')
 plt.show()
#Calculate field of view
def field_of_view(vol):
    #Image Shape in voxels
    n0,n1,n2=vol.shape
    #Sampling rate in mm
    d0,d1,d2=vol.meta['sampling']
    #field of view in mm
    f0,f1,f2=n0*d0,n1*d1,n2*d2
    return f0,f1,f2
#Extract Labels
def label_image(image):
 filt_m=ndi.median_filter(image,size=10)
 filt_g=ndi.gaussian_filter(filt_m,sigma=2)
 print(filt_g.dtype)
 print(filt_g.size)
 im_uint8=filt_g.astype(np.uint8)
 print(im_uint8.dtype)
 print(im_uint8.size)
 hist=ndi.histogram(filt_g,min=0,max=np.max(filt_g),bins=np.max(filt_g))
 print(hist.shape)
 plt.plot(hist)
 plt.show()
 mask = filt_g>400
 m=np.where(mask,image,0)
 eroted_mask=ndi.binary_erosion(m,iterations=5)
 dilated_mask=ndi.binary_dilation(eroted_mask,iterations=5)
 labels,nlabels=ndi.label(dilated_mask)
 boxes=ndi.find_objects(labels)
 return labels,nlabels,boxes,filt_m,filt_g,mask

#Plot images
def plot_image_comparison(image_1,image_2,image_3,image_4):
 fig,axes=plt.subplots(nrows=1,ncols=4)
 axes[0].imshow(image_1,cmap='gray')#General View
 axes[0].set_title('1st View')
 axes[0].axis('off')
 axes[1].imshow(image_2,cmap='gray')#cv image
 #axes[1].imshow(np.where(cv_labels==2,cv_image,1),cmap='rainbow',interpolation ='nearest', alpha = 0.1)#Coronal View
 axes[1].set_title('2nd View')
 axes[1].axis('off')
 axes[3].imshow(image_3,cmap='gray',alpha = 1, interpolation ='bilinear')#Sagital view
 #axes[3].imshow(cv_label_tumor_overlaped,cmap='gray',interpolation ='nearest', alpha = 0.8)#Coronal View
 axes[3].set_title('3rd View')
 axes[3].axis('off')
 axes[2].imshow(image_4,cmap='gray')#Sagital view
 axes[2].set_title('4th View')
 axes[2].axis('off')
 plt.show()
 #Set effective slices
def effective_slices(work_dataset):
  db_mean=work_dataset[89].mean()  #mean of centered slice in the dataset
  print(db_mean) #verify mean
  l_work_dataset=[] # empty list to store slices of interest   
  for i in range(0,len(work_dataset)):
      #Defining slices of interest based on the mean (range from 20 to 80% of mean)
      if ((work_dataset[i].mean() >= (db_mean*0.2)) & (work_dataset[i].mean() <= (db_mean*1.8))):
          l_work_dataset.append(work_dataset[i])#append slices of interest
  print(len(work_dataset))#Discovering number of slices selected
  np_wk_dataset=np.array(l_work_dataset)#convert list to numpy array
  print(np_wk_dataset.shape)# verify if quantity of slices are estill the same
  return np_wk_dataset