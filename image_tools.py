from PIL import Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import cv2
import camera
import motor
import shutil
import csv
import pandas as pd
import natsort

# Export image to jpeg file
def export_png(filename, image):
    print("Exporting image to '{}.png'".format(filename))
    image = Image.fromarray(image).convert('RGB')
    image.save(filename+".jpg")

# Import jpeg image as numpy array
def import_png(filename,greyscale=False):
    image = Image.open(filename)
    if greyscale:
        image = 0.299 * image[:,:,0] + 0.587 * image[0,0,1] + 0.114 * image[0,0,2]
    (height, width) = image.size
    print("Importing '{}' ({}x{})".format(filename,width,height))
    return np.array(image)

# Export list of numpy arrays to an output folder
def export_png_stack(foldername, images):
    print("Exporting images 'image(i).jpg' to folder '{}'".format(foldername))
    
    try:
        os.makedirs(foldername)
    except:
        pass

    for i,image in enumerate(images):
        fpath = "{}/image{}".format(foldername,i+1)
        export_png(fpath, image)

def import_png_stack(foldername, greyscale=False):
    print("Importing images from folder '{}'".format(foldername))

    # Get list of jpg/jpeg/png filenames
    files = glob.glob("{}/*.jpg".format(foldername))
    files.extend(glob.glob("{}/*.jpeg".format(foldername)))
    files.extend(glob.glob("{}/*.png".format(foldername)))
    files=natsort.natsorted(files)
    # Load files and append to images list
    images = []
    for file in files:
        image = np.array(Image.open(file))
        if greyscale:
            image = 0.299 * image[:,:,0] + 0.587 * image[0,0,1] + 0.114 * image[0,0,2]
        images.append(image)

    return images


                                ##Everything Will has added##

def take_z_stack(name, step, n): #Take a Z-Stack-specify file name, step increments, number of images
    cam=camera.get_camera()
    m=motor.Motor()
    i=0
    while i<=n*step:
          filename = name+str(i)+'.png'
          cam.capture(filename)
          m.steps(step)
          i += step

def get_z_stack(foldername, step, n):
    take_z_stack(foldername, step, n)
    parent_dir = '/media/pi/USB DISK/motorised-motic/part1'
    os.mkdir(parent_dir+'/Z-Stacks/'+foldername)
    z_stack_folder=parent_dir+'/Z-Stacks/'+foldername
    for file in os.listdir(parent_dir):
        if file.startswith(foldername):
           shutil.move(file, z_stack_folder)
#take a Z-Stack and move to Z-Stack folder
           
def get_fft(filename):
    print("Obtaining discrete fourier transform of '{}.png'".format(filename))
    img = cv2.imread(filename+".png", 0) #Import grayscale image
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
#Apply FastFourierTransform in 2D to the image
    dft = np.fft.fftshift(dft)
#Shift DC component from top left to center
    magnitude_spectrum = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
#Take magnitude to get the desired spectrum
    return(magnitude_spectrum)
#Obtain Fourier transform of given image


def plot_fft(filename):
    print("Plotting discrete fourier transform of '{}.png'".format(filename))
    magnitude_spectrum_scaled = 20 * np.log(get_fft(filename))
#Find the magnitude spectrum take log so not dominated by DC component.
    plt.imshow(magnitude_spectrum_scaled,cmap='gray')
    plt.title('FFT Image'), plt.xticks([]), plt.yticks([])
    plt.show()
#Obtain log plot of Fourier plot spectra

def get_mask(filename, filtertype, radius):
    print("Producing mask for'{}'".format(filename))
    img = cv2.imread(filename+".png", 0)
    #Circular HPF mask/filter, center circle of array is 0, remaining all ones
    rows, cols = img.shape #fit mask to dimension of image
    crow, ccol = int(rows / 2), int(cols / 2) 
    #Define centre of circle as centre of image (dims always even)
    mask = np.ones((rows, cols), np.uint8) #a 3D array of dimension rows x cols x 2
    center = [crow, ccol] #centre of circle
    x, y = np.ogrid[:rows, :cols]
    #x a column vector from 0, to number rows, y a row vector from 0 to number columns
    if filtertype == 'high':
         mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
    elif filtertype == 'low':
         mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= radius**2
      #Region in/outside of circle (disk of radius r)
    mask[mask_area] = 0 #set values in/outside of disk to 0
    return(mask)
#produce masks for high/low pass frequency filters in Fourier space - input filename, high/low filter,
#radius of circle defining filter

def get_cpf_mask(filename, r1, r2):
#should input r1<r2, else raise error
    mask1=get_mask(filename, 'high', r1)
    mask2=get_mask(filename, 'low', r2)
    mask=mask1*mask2 #matrix mult element-wise in python-get combined mask
    return(mask)
#get 'band' filter ie an annulous or ring of frequencies see above function

def apply_mask(filename, filtertype, radius):
    print("Applying mask for'{}'".format(filename))
    img = cv2.imread(filename+".png", 0)
    masked_img=img*(get_mask(filename, filtertype, radius))
#matrix mult element-wise in python-get combined mask
    return(masked_img)
#apply defined filters above to image

def apply_cpf_mask(filename, r1, r2):
    print("Applying CPF mask for'{}'".format(filename))
    img = cv2.imread(filename+".png", 0)
    masked_img=img*(get_cpf_mask(filename,r1,r2))
#matrix mult element-wise in python-get combined mask
    return(masked_img)
#apply band filter to image

def crop_image(filename):
    print("Cropping image'{}'".format(filename))
    img = cv2.imread(filename+'.png',0) #grayscale
    cropped_img=img[200:450, 260:600] #pre-determined dimensions
    return(cropped_img)
#cropping tool, may be useful for speed of blur algorithm if crop image to power of 2 dimensions for fast FFT

def crop_z_stack(foldername):
    print("Cropping Images in Folder'{}'".format(foldername))
    parent_dir = '/media/pi/USB DISK/motorised-motic/part1'
    dir = parent_dir+'/Z-Stacks/'+foldername
    z_stack = natsort.natsorted(os.listdir(dir))
    for file in z_stack:
        shutil.move(dir+'/'+file, parent_dir)
        file = file[:-4] #remove .png extension
        crop = crop_image(file)
        os.remove(parent_dir+'/'+file+'.png')
        cv2.imwrite(file+'.png',crop)
        shutil.move(parent_dir+'/'+file+'.png', dir)
#Crop a whole Z-Stack folder-black edge outside of Picamera distorts Fourier transform
        
def edge_detect(filename, lowthresh, highthresh):
    img = cv2.imread(filename+".png",0)
#Read Image, Convert to Grayscale(0)
    edges = cv2.Canny(img, lowthresh, highthresh)
#Detect only the edges i.e pixels above thresholds and if between thresholds, 
#connected to an edge above threshold
    return(edges)
#Canny edge detection

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
#split array in rectangular sub arrays of size p,q

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))
#get original array from rectangular sub arrays

def detail_map(filename, p, q):
#ensure that k,m divide respective dimensions of image
    edge_img=edge_detect(filename,20,60) #edges map
    rows, cols = edge_img.shape
    block_img = blockshaped(edge_img, p, q)
    blocks = block_img.shape[0]
    for i in range(blocks):
        density=np.count_nonzero(block_img[i,:,:])/(p*q) #density formula for each block
        detail=255*density #get an intensity map relative to edge density
        block_img[i,:,:]=detail*np.ones((p,q)) #Change block to constant block of pixel intensity
    detail_map=unblockshaped(block_img, rows, cols)
    return(detail_map)
#Produce a detail map of edge densities in each rectangular sub array

def get_detail_stats(filename, p, q):
#ensure that k,m divide respective dimensions of image
    edge_img = edge_detect(filename,20,60) #edges map
    rows, cols = edge_img.shape
    block_img = blockshaped(edge_img, p, q)
    blocks = block_img.shape[0]
    local_densities=np.array([]) #record local detail of each block
    for i in range(blocks):
        density=np.count_nonzero(block_img[i,:,:])/(p*q) #density formula for each block
        local_densities=np.append(local_densities,density)
    mu=np.amax(local_densities) #maximum density value
    rho=np.mean(local_densities) #average density value
    #sparse_rho=np.mean(local_densities[local_densities != 0]) #average of non-zero density values
    return({'Max Density':mu,'Avg. Density':rho})
#give parameters used in detail metric

def get_detail_metric(filename, p, q):
    stats=get_detail_stats(filename, p, q)
    mu=stats['Max Density']
    rho=stats['Avg. Density']
    delta=(rho*np.exp((mu-1)/(mu+0.0000001)))**(1/10) #1/10, emperically derived, 0.001-avoid div by zero for no detail images
    return(delta)
#gives detail metric normalised to [0,1], increasing with detail

def get_ring_masks(filename, n):
    print("Obtaining available disks for'{}'".format(filename))
    img = cv2.imread(filename+".png", 0)
    r, c = img.shape
    r, c = int(r/2), int(c/2)
    max_rad = np.amin(np.array([r, c])) 
    C_0_rad = int(max_rad/np.sqrt(n)) #Split into even area disks
    C_0=get_mask(filename, 'low', C_0_rad)
    ring_masks=np.array(C_0,ndmin=3)
    for i in range(1,n):
        r1=int(np.sqrt(i)*C_0_rad)
        r2=int(np.sqrt(i+1)*C_0_rad)
        ring_i=get_cpf_mask(filename,r1,r2)
        ring_i=np.array(ring_i,ndmin=3)
        ring_masks=np.append(ring_masks,ring_i,axis=0)
    return(ring_masks)
#returns 3D array of each ring mask indexed by ring_masks[0,:,:], [1,:,:], etc, for concentric fourier rings

def get_fourier_energies(filename, n): 
    fourier =  get_fft(filename)
    ring_masks = get_ring_masks(filename, n) 
    energies = np.array([])
    cumulative_energies = np.array([])
    for j in range(n):
        ring_j = ring_masks[j,:,:]
        fourier_ring = fourier*ring_j 
        ring_energies = fourier_ring[fourier_ring != 0]
        ring_energy = np.sum(ring_energies)
        energies = np.append(energies,ring_energy)
        cumulative_energy_j = np.sum(energies)
        cumulative_energies = np.append(cumulative_energies,cumulative_energy_j)
    ratio_energies = np.array([])
    for j in range(n):
        ratio_energy_j = cumulative_energies[j]/cumulative_energies[n-1]
        ratio_energies=np.append(ratio_energies, ratio_energy_j)
    return({'Ring Energies':energies, 'Cumulative Energies':cumulative_energies, 'Energy Ratios':ratio_energies})
#gives blur parameter values
def get_energy_hist(filename, n):
    energy=get_fourier_energies(filename ,n)
    ratios=energy['Energy Ratios']
    R_0=ratios[0]
    Ring_j=np.array(range(1,n+1))
    plt.bar(Ring_j, ratios)
    plt.title('Cumulative Frequency Ratios')
    plt.xlabel('j-th Ring')
    plt.ylabel('R_n')
    plt.plot([1,n], [R_0,1], color='red')
    plt.show()
#produce fourier ring energyies histogram
    
def get_gaussian_blur(filename, n):
#n should be positive and odd
    img=cv2.imread(filename+".png",0)
    gaussian_img=cv2.GaussianBlur(img, (n,n), cv2.BORDER_DEFAULT)
    return(gaussian_img)
#artifical blur

def get_blur_stats(filename, n):
    energies=get_fourier_energies(filename, n)
    ratios=energies['Energy Ratios']
    R_0 = ratios[0]
    R = np.sum(ratios)
    Delta = (n*(1+R_0))/(2*R)
    return({'Initial Ring Energy R_0':R_0,'Sum of Cumulative Ring Energy Ratios':R, 'Idealised No Blur Estimate':Delta})
#blur parameter values
def get_blur_metric(filename, n):
    stats=get_blur_stats(filename, n)
    R_0=stats['Initial Ring Energy R_0']
    R=stats['Sum of Cumulative Ring Energy Ratios']
    Delta=stats['Idealised No Blur Estimate']
    blur=((Delta)*np.exp((R_0-1/(R_0+0.00001))))**(1/3)
    return(blur)
#blur metric value, normalised to [0,1], decreasing with focus, increasing with blur

def get_z_stack_metrics(foldername, step, n, p, q):
    parent_dir = '/media/pi/USB DISK/motorised-motic/part1/Z-stacks/'+foldername
    z_stack = natsort.natsorted(os.listdir(parent_dir))
    number_files = len(z_stack)
    details = []
    blurs = []
    hybrids = []
    for file in z_stack:
        shutil.move(parent_dir+'/'+file,'/media/pi/USB DISK/motorised-motic/part1')
        file = file[:-4] #get rid of .png on end of file
        d = get_detail_metric(file, p, q)
        b = get_blur_metric(file, n)
        h = get_hybrid_metric(file, n, p, q)
        details = np.append(details, d)
        blurs = np.append(blurs, b)
        hybrids = np.append(hybrids, h)
        shutil.move('/media/pi/USB DISK/motorised-motic/part1/'+file+'.png', parent_dir)
    metrics = np.transpose(np.vstack((details, blurs, hybrids)))
    data=pd.DataFrame(metrics, columns=['Details', 'Blurs', 'Hybrids'])
    index=np.linspace(0, step*(number_files-1), number_files, dtype=int)
    data.index=index.tolist()
    data.index.name='steps'
    data.to_csv(r'/media/pi/USB DISK/motorised-motic/part1/Z-stacks/'+foldername+'/'+foldername+'data.csv')      
    return(z_stack)
#write metric values to a csv file, saved to the z-stack file for metric analytics and emperical derivations