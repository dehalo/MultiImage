#from PIL import Image, ImageFilter
from tkinter import filedialog
import cv2
import numpy as np
import statistics

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

"https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/"

def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
  return im1Reg, h
 
 
if __name__ == '__main__':
#AM
  n=int(input('#of images: '))
  im=[]
  for i in range(0,n):
   file = filedialog.askopenfile(mode="r", title="select file", filetypes=(("jpgs", "*.jpg"), ("all files", "*.*"))) #initialdir="/", gelöscht, 2. Position
   print(file.name)
   im.append(cv2.imread(file.name, cv2.IMREAD_COLOR))
   if (i==0):
    outFilename=file.name[0:len(file.name)-4]+"_Mult"+str(n)+".jpg"
   
  imReference = im[0]
  for i in range(1,n):
   print("Aligning images ...")
   # Registered image will be resotred in imReg. 
   # The estimated homography will be stored in h. 
   im[i], h = alignImages(im[i],imReference)
   # Print estimated homography
   print("Estimated homography : \n",  h)
   #cv2.destroyAllWindows()
   #cv2.imshow('image',im[i])
   #cv2.imshow('image',im[i])
   #cv2.waitKey(0)
   #
  
  # px=[256,300] #Versuch, ihn zu 2 Byte zu zwingen hilft nicht
  h, w, c = im[0].shape #BGR
  for i in range (0,h):
   for j in range (0,w):
    #bk=True
    pb=[]
    pg=[]
    pr=[]
    diff=[]
    for l in range (0,n):
      if ((im[l][i,j][0] != 0) or (im[l][i,j][1] != 0) or (im[l][i,j][2] != 0)):
       pb.append(im[l][i,j][0]+256)#sonst verwendet Median Byte und produziert overflow
       pg.append(im[l][i,j][1]+256)
       pr.append(im[l][i,j][2]+256)
    if (pb!=[]):
     medb=statistics.median(pb)-256
    else:
     medb=0
    if (pg!=[]):
     medg=statistics.median(pg)-256
    else:
     medg=0
    if (pr!=[]):
     medr=statistics.median(pr)-256
    else:
     medr=0
    for l in range (0,n):
      if ((im[l][i,j][0] != 0) or (im[l][i,j][1] != 0) or (im[l][i,j][2] != 0)):
        diff.append(abs(im[l][i,j][0]-medb)+abs(im[l][i,j][1]-medg)+abs(im[l][i,j][2]-medr)*2) #da müssten wir doch auch die 0,0,0 ausschliessen bzw Null setzen, 2 um rot überzugewichten (Gesichter, Mond)
      else:
        diff.append(-1)
    for l in range (1,n): #allenfalls 1,n, minimalWert für diff festlegen, sonst im[0] unverändert lassen
      if (diff[l]==max(diff)) and (diff[l]>10):
       im[0][i,j]=im[l][i,j]
       #bk= False #Basis Median statt im[0], dann muss for Schlaufe 0,n sein statt 1,n
      #if bk:
        #im[0][i,j][0]=medb 
        #im[0][i,j][1]=medg
        #im[0][i,j][2]=medr
        
       
# Write aligned image to disk. 
  #outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, im[0])
  
"""AM, https://pillow.readthedocs.io/en/3.0.x/reference/PixelAccess.html
for i in range(0,n):
 im[i].show()
 px=im[i].load()
 print(px[4,4])
 px[4,4] = (0,0,0)"""
                   