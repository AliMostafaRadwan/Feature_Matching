# import streamlit as st
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# def calculate_diff_ratio(img1, img2):
#     orb = cv2.ORB_create(nfeatures = 5000 , scoreType=cv2.ORB_FAST_SCORE)

#     kp1, des1 = orb.detectAndCompute(img1,None)
#     kp2, des2 = orb.detectAndCompute(img2,None)

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#     matches = bf.match(des1,des2)
#     matches = sorted(matches, key = lambda x:x.distance)

#     diff_ratio = len(matches) / min(len(kp1), len(kp2))

#     return diff_ratio



# def mse(imageA, imageB):
# 	# the 'Mean Squared Error' between the two images is the
# 	# sum of the squared difference between the two images;
# 	# NOTE: the two images must have the same dimension
# 	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
# 	err /= float(imageA.shape[0] * imageA.shape[1])
	
# 	# return the MSE, the lower the error, the more "similar"
# 	# the two images are
# 	return err

# img1 = cv2.imread('images/nss2.jpeg',0)
# img2 = cv2.imread('images/nss2.jpeg',0)
# (thresh, img_bin) = cv2.threshold(img1, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

# #invert the image
# img_bin = 255-img_bin

# (thresh, img_bin2) = cv2.threshold(img2, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# img_bin2 = 255-img_bin2


# # diff_ratio = calculate_diff_ratio(img_bin, img_bin2)
# diff_ratio = calculate_diff_ratio(img1, img2)


# st.title("Image Matcher and Difference Ratio Calculator")

# st.header("Image Matcher")
# col1 ,col2 = st.columns(2)
# col1.header("Image 1")
# col2.header("Image 2")

# col1.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption="Image 1", use_column_width=True)
# col2.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), caption="Image 2", use_column_width=True)

# orb = cv2.ORB_create()

# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# matches = bf.match(des1,des2)
# matches = sorted(matches, key = lambda x:x.distance)

# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None, flags=2)
# st.image(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB), caption="Matches", use_column_width=True)

# st.header("Difference Ratio Calculator")
# st.write("The difference ratio is:", diff_ratio)
# err = mse(img_bin, img_bin2)
# st.write("The mean squared error is:", err/1000)




import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load the two images

img1 = cv2.imread('images/ty.jpeg')
img2 = cv2.imread('images/ty2.jpeg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Approach 1: Pixel-wise comparison
diff = cv2.absdiff(gray1, gray2)
diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

# Approach 2: Histogram comparison
hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Approach 3: Edge detection
edges1 = cv2.Canny(gray1, 100, 200)
edges2 = cv2.Canny(gray2, 100, 200)
score = ssim(edges1, edges2)

# Approach 4: Template matching
template = cv2.imread('images/nss3.jpeg', 0)
res = cv2.matchTemplate(gray1, template, cv2.TM_CCOEFF_NORMED)
match_score = cv2.minMaxLoc(res)[1]

# Print the results
print('Pixel-wise difference:', np.mean(diff))
print('Histogram correlation:', corr)
print('SSIM score:', score)
print('Template matching score:', match_score)
