import numpy as np
from scipy import ndimage
from scipy import spatial
from scipy import io
from scipy import sparse
from scipy.sparse import csgraph
from scipy import linalg
from matplotlib import pyplot as plt
# import seaborn as sns
from skimage import data
from skimage import color
from skimage import img_as_float
from skimage.measure import compare_ssim
from scipy.stats import wasserstein_distance
from scipy.ndimage import imread
from skimage.transform import resize
import cv2
import matplotlib
# from image_similarity import structural_sim, pixel_sim, sift_sim, earth_movers_distance


# import graph3d
# %matplotlib inline




def draw_image(image):
    fig, ax = plt.subplots()
    plt.imshow(image, cmap='gray')
    plt.grid('off')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_title('Original image')
    plt.savefig('./tikhonov_regularization_0.png', bbox_inches='tight')


def crop_add_noise(image):
    # image = image[40:80, 100:140]
    image = image[40:80, 100:140]
    noisy_image = image + 0.05 * np.random.randn(*image.shape)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(noisy_image, cmap='gray')
    ax[0].grid('off')
    ax[1].grid('off')
    ax[0].xaxis.set_ticks([])
    ax[0].yaxis.set_ticks([])
    ax[1].xaxis.set_ticks([])
    ax[1].yaxis.set_ticks([])
    ax[0].set_title('Cropped image')
    ax[1].set_title('Noisy image')
    plt.savefig('./tikhonov_regularization_1.png', bbox_inches='tight')
    return image, noisy_image


def plot_save(image, outpath, filename):
    fig, ax = plt.subplots()
    plt.imshow(image, cmap='gray')
    plt.grid('off')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_title(filename[:-4])
    plt.savefig(outpath+filename, bbox_inches='tight')


def plot_output(image, noisy_image, graph_filtered_image, traditional_filtered_image):
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))

    ax.flat[0].imshow(image, cmap='gray')
    ax.flat[1].imshow(noisy_image, cmap='gray')
    ax.flat[2].imshow(graph_filtered_image, cmap='gray')
    ax.flat[3].imshow(traditional_filtered_image, cmap='gray')
    ax.flat[0].grid('off')
    ax.flat[1].grid('off')
    ax.flat[2].grid('off')
    ax.flat[3].grid('off')
    ax.flat[0].xaxis.set_ticks([])
    ax.flat[0].yaxis.set_ticks([])
    ax.flat[1].xaxis.set_ticks([])
    ax.flat[1].yaxis.set_ticks([])
    ax.flat[2].xaxis.set_ticks([])
    ax.flat[2].yaxis.set_ticks([])
    ax.flat[3].xaxis.set_ticks([])
    ax.flat[3].yaxis.set_ticks([])
    ax.flat[0].set_title('Cropped Image')
    ax.flat[1].set_title('Noisy Image')
    ax.flat[2].set_title('Graph Filtered')
    ax.flat[3].set_title('Gaussian Filtered')
    plt.tight_layout()
    plt.savefig('./tikhonov_regularization_2.png', bbox_inches='tight')


def query_neighbor_pixels(noisy_image, theta, kappa):
    tmp = np.vstack(np.dstack(np.indices(noisy_image.shape)))
    tree = spatial.cKDTree(tmp)  # kd-tree for quick nearest-neighbor lookup
    q = tree.query_ball_point(tmp, kappa)  # find all points within distance r of points
    I = np.concatenate([np.repeat(k, len(q[k])) for k in range(len(q))])
    J = np.concatenate(q)

    # Distance metric is difference between neighboring pixels
    dist_ij = np.sqrt((noisy_image.flat[I] - noisy_image.flat[J]) ** 2)
    # Gaussian kernel weighting function, threshold
    W = np.exp(- ((dist_ij) ** 2 / 2 * (theta ** 2)))
    print('done')
    return I, J, W


def gen_Adjacency_matrix(noisy_image, I, J, W):
    A = sparse.lil_matrix((noisy_image.size, noisy_image.size))
    for i, j, w in zip(I, J, W):
        A[i, j] = w
        A[j, i] = w

    A = A.todense()
    return A


def Compute_Laplacian(A, gamma):
    L = csgraph.laplacian(A)
    l, u = linalg.eigh(L)  # compute eigenvalues and eigenvectors of laplacian
    h = u @ np.diag(1 / (1 + gamma * l)) @ u.T  # Compute filtering kernel
    graph_filtered_image = (h @ noisy_image.ravel()).reshape(noisy_image.shape)

    # Filter the image using traditional gaussian filtering
    traditional_filtered_image = ndimage.gaussian_filter(noisy_image, sigma=1.5)
    # traditional_filtered_image = ndimage.gaussian_filter(noisy_image, sigma=3.5)

    return graph_filtered_image, traditional_filtered_image


def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w)


def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    img = img.astype(int)
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i + 1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)


def get_img(path, norm_size=True, norm_exposure=False):
  '''
  Prepare an image for image processing tasks
  '''
  # flatten returns a 2d grayscale array
  img = imread(path, flatten=True).astype(int)
  # resizing returns float vals 0:255; convert to ints for downstream tasks
  height,width = img.shape

  if norm_size:
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
  if norm_exposure:
    img = normalize_exposure(img)
  return img

def earth_movers_distance(path_a, path_b):
  '''
  Measure the Earth Mover's distance between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  hist_a = get_histogram(img_a)
  hist_b = get_histogram(img_b)
  return wasserstein_distance(hist_a, hist_b)


def PSNR(path_a, path_b):
    img1 = cv2.imread(path_a)
    img2 = cv2.imread(path_b)
    psnr = cv2.PSNR(img1, img2)
    return psnr

def structural_sim(path_a, path_b):
  '''
  Measure the structural similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  img_a = get_img(path_a)
  img_b = get_img(path_b)
  sim, diff = compare_ssim(img_a, img_b, full=True)
  return sim


def pixel_sim(path_a, path_b):
  '''
  Measure the pixel-level similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''

  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  height , width = img_a.shape
  return np.sum(np.absolute(img_a - img_b)) / (height*width) / 255


def sift_sim(path_a, path_b):
  '''
  Use SIFT features to measure image similarity
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  # initialize the sift feature detector
  orb = cv2.ORB_create()

  # get the images
  img_a = cv2.imread(path_a)
  img_b = cv2.imread(path_b)

  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)

  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 70]
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)


def similarity(path_a, path_b):
    psnr = PSNR(path_a, path_b)
    structural_simi = structural_sim(path_a, path_b)
    pixel_simi = pixel_sim(path_a, path_b)
    sift_simi = sift_sim(path_a, path_b)
    emdi = earth_movers_distance(path_a, path_b)
    print(psnr)
    print(structural_simi)
    print(pixel_simi)
    print(sift_simi)
    print(emdi)


if __name__ == '__main__':
    # global structural_sim, pixel_sim, sift_sim
    kappa = np.sqrt(2)
    # kappa = 0
    theta = 20
    gamma = 10

    # orig_image = data.coffee()

    # orig_image = data.camera()
    orig_image = data.page()
    # orig_image = data.coins()

    image = img_as_float(orig_image[::2, ::2])
    draw_image(image)
    cropped_image, noisy_image = crop_add_noise(image)
    I, J, W = query_neighbor_pixels(noisy_image, theta, kappa)
    A = gen_Adjacency_matrix(noisy_image, I, J, W)
    graph_filtered_image, traditional_filtered_image = Compute_Laplacian(A, gamma)
    plot_output(cropped_image, noisy_image, graph_filtered_image, traditional_filtered_image)
    #
    # plot_save(cropped_image, './', 'crop_image.jpg')
    # plot_save(noisy_image, './', 'crop_noisy_image.jpg')
    # plot_save(graph_filtered_image, './', 'graph_filtered_image.jpg')
    # plot_save(traditional_filtered_image, './', 'traditional_filtered_image.jpg')

    # path_crop = 'crop_image.jpg'
    # path_noisy = 'crop_noisy_image.jpg'
    # path_graph = 'graph_filtered_image.jpg'
    # path_trad = 'traditional_filtered_image.jpg'
    # print('similarity cropped image V.S. cropped noisy image')
    # similarity(path_crop, path_noisy)
    #
    # print('similarity cropped image V.S. graph filtered image')
    # similarity(path_crop, path_graph)
    #
    # print('similarity cropped image V.S. traditional filtered image')
    # similarity(path_crop, path_trad)
