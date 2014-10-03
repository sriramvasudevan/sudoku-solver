# built-in modules
from multiprocessing.pool import ThreadPool

import cv2

import numpy as np
from numpy.linalg import norm

# local modules
from common import clock, mosaic



SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 10
DIGITS_FN = 'digits.png'

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_digits(fn):
    print 'loading "%s" ...' % fn
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (SZ, SZ))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return digits, labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print 'confusion matrix:'
    print confusion
    print

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    return mosaic(25, vis)

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

def generate_mosaic(directory):
    digits = np.empty([0,SZ,SZ])
    prefix = str(SZ)+'/cropped_img'
    suffix = '.png'
    for i in range(1,11):
        for j in range(20,100):
            infix= '%03d-%05d' % (i, j)
            fname = directory+prefix+infix+suffix
            img = cv2.imread(fname, 0)
            digits = np.append(digits, np.array([img]),0)
    cv2.imwrite('font'+str(SZ)+'.png', mosaic(20, digits))

if __name__ == '__main__':
    generate_mosaic('cropped')
    #digits, labels = load_digits(DIGITS_FN)
    digits, labels = load_digits('font'+str(SZ)+'.png')
    #fonts, fontlabels = load_digits('font.png')
    #digits = np.concatenate((digits, fonts), axis=0)
    #labels = np.concatenate((labels, fontlabels), axis=0)
    print digits.shape, labels.shape

    print 'preprocessing...'
    # shuffle digits
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    digits2 = map(deskew, digits)
    samples = preprocess_hog(digits2)

    train_n = int(0.9*len(samples))
    #train_n = int(1.0*len(samples))
    cv2.imshow('train set', mosaic(25, digits[:train_n]))
    digits_train, digits_test = np.split(digits2, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])


    digits_img_test = cv2.imread('final'+str(SZ)+'.png', 0)
    digits_img_test = cv2.bitwise_not(digits_img_test)
    digits_test = split2d(digits_img_test, (SZ, SZ))
    digits2_test = map(deskew, digits_test)
    samples_test = preprocess_hog(digits2_test)
    cv2.imshow('test set', mosaic(9, digits_test))
    #cv2.imshow('test set', mosaic(25, digits_test))

    print 'training SVM...'
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    #vis = evaluate_model(model, digits_test, samples_test, labels_test)
    #cv2.imshow('SVM test', vis)
    results = model.predict(samples_test)
    #print results
    for i in range(0,81,9):
        print results[i:i+9]
    print 'saving SVM as "digits_svm.dat"...'
    model.save('digits_svm.dat')

    cv2.waitKey(0)

