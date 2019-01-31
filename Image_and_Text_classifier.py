
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image,ImageDraw
from scipy.ndimage.filters import rank_filter
import sys,argparse
from pytesseract import image_to_string
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC,NuSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import os


# In[2]:

def image_spam_score(image,model):
    image_data = tf.gfile.FastGFile(image,'rb').read()
    
    with tf.gfile.FastGFile(model,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name='')
    
    with tf.Session() as sees:
        softmax_tensor = sees.graph.get_tensor_by_name('final_result:0')
        predictions = sees.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})
    
    ham = predictions[0][0] * 100
    spam = predictions[0][1] * 100
    return ham,spam
    


# In[3]:

def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)

def pad_crop(crop, contours, edges, border_contour, pad_px=15):
    """Slightly expand the crop to get full contours.
    This will expand to include any contours it currently intersects, but will
    not expand past a border.
    """
    bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
    if border_contour is not None and len(border_contour) > 0:
        c = props_for_contours([border_contour], edges)[0]
        bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

    def crop_in_border(crop):
        x1, y1, x2, y2 = crop
        x1 = max(x1 - pad_px, bx1)
        y1 = max(y1 - pad_px, by1)
        x2 = min(x2 + pad_px, bx2)
        y2 = min(y2 + pad_px, by2)
        return crop
    
    crop = crop_in_border(crop)

    c_info = props_for_contours(contours, edges)
    changed = False
    for c in c_info:
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        this_area = crop_area(this_crop)
        int_area = crop_area(intersect_crops(crop, this_crop))
        new_crop = crop_in_border(union_crops(crop, this_crop))
        if 0 < int_area < this_area and crop != new_crop:
            print ('%s -> %s' % (str(crop), str(new_crop)))
            changed = True
            crop = new_crop

    if changed:
        return pad_crop(crop, contours, edges, border_contour, pad_px)
    else:
        return crop

def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def crop_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0))/255
        })
    return c_info


def find_optimal_components_subset(contours, edges):
    """Find a crop which strikes a good balance of coverage/compactness.
    Returns an (x1, y1, x2, y2) tuple.
    """
    c_info = props_for_contours(contours, edges)
    c_info.sort(key=lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area = edges.shape[0] * edges.shape[1]

    c = c_info[0]
    del c_info[0]
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    while covered_sum < total:
        changed = False
        recall = 1.0 * covered_sum / total
        prec = 1 - 1.0 * crop_area(crop) / area
        f1 = 2 * (prec * recall / (prec + recall))
        #print '----'
        for i, c in enumerate(c_info):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            new_crop = union_crops(crop, this_crop)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * crop_area(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

            # Add this crop if it improves f1 score,
            # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
            # ^^^ very ad-hoc! make this smoother
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
            if new_f1 > f1 or (
                    remaining_frac > 0.25 and new_area_frac < 0.15):
                print ('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
                        i, covered_sum, new_sum, total, remaining_frac,
                        crop_area(crop), crop_area(new_crop), area, new_area_frac,
                        f1, new_f1))
                crop = new_crop
                covered_sum = new_sum
                del c_info[i]
                changed = True
                break

        if not changed:
            break

    return crop


def dilate(ary, N, iterations): 
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""
    kernel = np.ones((N,N), dtype=np.uint8)
    kernel[(N-1)//2,:] = 1
    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

    kernel = np.ones((N,N), dtype=np.uint8)
    kernel[:,(N-1)//2] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image


def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.
    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    count = 21
    dilation = 5
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilate(edges, N=3, iterations=n)
        dilated_image = dilated_image.astype(np.uint8)
        ret,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    return contours


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))

def remove_border(contour, ary):
    """Remove everything outside a border contour."""
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)


def find_border_components(contours,ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i,c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w*h > 0.5 * area:
            borders.append((i,x,y,x+w-1,y+h-1))
    return borders


def downscale(im,max_dim=1024):
    a,b = im.size
    if max(a,b) <= max_dim:
        return 1.0,im
    scale = 1.0 * max_dim / max(a,b)
    new_image = im.resize((int(scale*a),int(scale*b)),Image.ANTIALIAS)
    return scale,new_image



# In[4]:

def crop_text_from_image(image):
    image = Image.open(image)
    
    scale,new_image = downscale(image)
    edges = cv2.Canny(np.asarray(new_image),100,200)
    ret,contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    borders = find_border_components(contours,edges)
    borders.sort(key=lambda i_x1_y1_x2_y2: (i_x1_y1_x2_y2[3] - i_x1_y1_x2_y2[1]) * (i_x1_y1_x2_y2[4] - i_x1_y1_x2_y2[2]))
    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)
    
    edges = 255 * (edges > 0).astype(np.uint8)
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered
    
    contours = find_components(edges)
    if len(contours) == 0:
        print ('%s -> (no text!)' % path)
        return image
    
    crop = find_optimal_components_subset(contours, edges)
    crop = pad_crop(crop, contours, edges, border_contour)
    crop = [int(x / scale) for x in crop]
    
    text_im = image.crop(crop).convert('RGB')
    return text_im


# In[5]:

def text_image_processing(text_image):#PIL image is taken as input
    image = np.array(text_image)
    image = image[:,:,::-1] #Converting RGB to BGR
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,3,21,21)
    return gray
    


# In[6]:

def rem_stopwords(text):
    stopwords = open("stopwords.txt","r",encoding="utf8")
    stwords=stopwords.read()

    words = re.sub('\n',' ',stwords)
    word = words.split()
    clearW = [wor for wor in text if wor not in word]
    return clearW


# In[7]:

def hi_stem(lst):
    
    suffixes = {
    1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
    2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें","ाना",
        "ाऊ","लु","ाव","ीय","हट"],
    3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं",
        "ाएं", "ुओं", "ुएं", "ुआं","दार","िया","हार","ावट","ोला","कार","ान","ौटी","ैया","ेरा","ोड़ा"],
    4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं",
        "ताएं", "ियाँ", "ियों", "ियां","वाला","िक"],
    5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां","क्कड़"]
    }

    
    txt = []
    for word in lst: 
        for L in 5, 4, 3, 2, 1:
            if len(word) > L + 1:
                for suf in suffixes[L]:
                    if word.endswith(suf):
                        word = word[:-L]  
        txt.append(word)      
    return txt


# In[8]:

def purify(doc):
    punctuations = '''!@#$£%^&*()"':-+_=“”/?><|।.,\{}[]'''
    digits = '''1234567890०१२३४५६७८९'''
    data_no_punc = ""
    for char in doc:
        if(char not in punctuations and char not in digits):
            data_no_punc = data_no_punc + char
    return data_no_punc


# In[16]:

def processed_text(text):
    no_punct = purify(text)
    no_stopwords = rem_stopwords(no_punct.split())
    stemmed = hi_stem(no_stopwords)
    return " ".join(stemmed)
    


# In[20]:

def text_spam(text):
    raw_data = pd.read_excel("hindi_spam.xlsx")
    E_mails = raw_data
    i=0
    for e in E_mails['text']:
        E_mails.text[i]=''.join(list(map(purify,e)))
        E_mails.text[i]=E_mails.text[i].split()
        i=i+1

    E_mails['text']=list(map(rem_stopwords,E_mails['text']))

    email = []
    email = (list(map(hi_stem,E_mails['text'])))
    E_mails['text'] = email

    for  i in range(0,E_mails.shape[0]):
        E_mails['text'][i] = ' '.join(E_mails['text'][i])

    transformer2 = TfidfVectorizer(ngram_range=(1,1))
    counts2 = transformer2.fit_transform(E_mails['text'])

    NBModel = BernoulliNB().fit(counts2, E_mails['type'])
    SVCModel = SVC(kernel='linear').fit(counts2,E_mails['type'])
    NuSVCModel = NuSVC(kernel='linear').fit(counts2,E_mails['type'])
    RFModel = RandomForestClassifier(n_estimators=50,min_samples_split=3).fit(counts2,E_mails['type'])
    GBModel = GradientBoostingClassifier(n_estimators=50,min_samples_split=200).fit(counts2,E_mails['type'])
    
    counts1 = transformer2.transform([text])
    
    NBpred = NBModel.predict(counts1)
    SVCpred = SVCModel.predict(counts1)
    NuSVCpred = NuSVCModel.predict(counts1)
    RFpred = RFModel.predict(counts1)
    GBpred = GBModel.predict(counts1)
    
    pred_list = [NBpred,SVCpred,NuSVCpred,RFpred,GBpred]
    pred = max(pred_list,key=pred_list.count)
    
    
    return pred[0]


# In[11]:

def image_classification(image,model):
    ham,spam = image_spam_score(image,model)
    img = Image.open(image).convert('RGB')
    cv_image = np.array(img)
    cv_image = cv_image[:,:,::-1]

    

    if spam > 80:
        cv2.imwrite(os.path.join('mails/spam/images/',"spam.png"),cv_image)
        print("SPAM!!!!")
        
    else:
        text_image = crop_text_from_image(image)
        text_image = text_image_processing(text_image)
        text = image_to_string(text_image,lang='hin')
        text = processed_text(text)
        pred = text_spam(text)
    
        if pred == 'spam':
            cv2.imwrite(os.path.join('mails/spam/images/','spam.png'),cv_image)
            print("SPAM!!!!!")
        else:
            cv2.imwrite(os.path.join('mails/ham/images/',"ham.png"),cv_image)
            print("HAM!!!!")


# In[12]:

def text_classification(text):
#     with open(text,'r') as f:
#         text = f.read()
    text1 = processed_text(text)
    pred = text_spam(text1)
    if pred == 'spam':
        with open(os.path.join('mails/spam/text/','spam.txt'),'w') as f:
            f.write(text)
        print("SPAM!!!")
    else:
        with open(os.path.join('mails/ham/text/','ham.txt'),'w') as f:
            f.write(text)
        print("HAM!!!")


# In[13]:

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--image',required = False,help = "Path to the image")
    ap.add_argument('-m','--model',required = False,help = "path to the model")
    ap.add_argument('-t','--text',required = False,help = "Path to the text file")
#     ap.add_argument('-s','--spam',required = True,help = "Path to the Spam Directory")
#     ap.add_argument('-ha','--ham',required = True,help = "Path to the Ham directory")
    args = vars(ap.parse_args())

#     spam_path = args['spam']
#     ham_path = args['ham']

    if args['text'] is not None:
        with open(args['text'],'r') as f:
            text = f.read()
        text_classification(text)
        
    elif args['image'] is not None:
        im = args['image']
        model = args['model']
        image_classification(im,model)
    else:
        print("please provide proper arguments for help type -h or --help")

    

