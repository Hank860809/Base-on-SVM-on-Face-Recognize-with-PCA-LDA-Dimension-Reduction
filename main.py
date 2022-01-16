# Hackmd:https://hackmd.io/9X_MuHSERz-_sfctp7ThJw
# github:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def show_img(figure_num, n,img_set):
    fig = plt.figure(figure_num)
    for i in range(n):
        tmp_img = img_set[i * 5].reshape(112,92)
        plt.subplot(int(np.ceil(n/5)),5,i+1)
        plt.imshow(tmp_img,cmap='gray')
    plt.show()

train_data = []
test_data = []
train_true = [] # 真實訓練資料標籤
test_true = []  # 真實訓練資料標籤
train_img = []  # 訓練集
test_img = []   # 測試集

# 資料預處理:將資料1~5當成訓練資料用的圖片，6~10作為訓練用的圖片
for i in range(1,41):
    for j in range(1,11):
        if j <= 5:
            train_data.append(plt.imread('att_faces/s{}/{}.pgm'.format(i,j), -1))
            train_true.append(i)
        else:
            test_data.append(plt.imread('att_faces/s{}/{}.pgm'.format(i,j), -1))
            test_true.append(i)

# 將訓練資料轉成一維陣列
for image in train_data:
    img_to_1D = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            img_to_1D.append(image[x][y])
    train_img.append(img_to_1D)

# 將測試資料轉成一維
for image in test_data:
    img_to_1D = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            img_to_1D.append(image[x][y])
    test_img.append(img_to_1D)

# 轉成陣列形式
train_img = np.array(train_img)
test_img = np.array(test_img)
train_true = np.array(train_true)
test_true = np.array(test_true)

# param_grid 為GridSearchCV 預設參數
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
pca_clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'),param_grid)
lda_clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'),param_grid)

# PCA 10,20,30,40,50維訓練
for i in range(10,51,10):
    print()
    print('第{}維:'.format(i))
    myPCA = PCA(n_components=i).fit(train_img)
    pca_train_img = myPCA.transform(train_img)
    pca_test_img = myPCA.transform(test_img)

    pca_pred = pca_clf.fit(pca_train_img, train_true)
    pca_test_pred = pca_pred.predict(pca_test_img)
    pca_result = confusion_matrix(test_true, pca_test_pred)

    # 準確率
    pca_accuracy = str(accuracy_score(pca_test_pred,test_true)*100)+"%"
    pca_report = classification_report(test_true,pca_test_pred)
    # 列印混合矩陣
    # print('PCA混淆矩陣:')
    # for j in range(len(pca_result)):
    #     for k in range(len(pca_result[j])):
    #         print(pca_result[j][k],end=" ")
    #     print()
    # 計算正確預測個數
    true_count = 0
    for j in range(len(pca_result)):
        true_count += pca_result[j][j]
    # 錯誤預測個數
    false_count = 200 - true_count
    # 統計資料
    print('PCA正確預測個數:{}  錯誤預測個數:{} 準確率:{}'.format(true_count,false_count,pca_accuracy))
    # 列印混合矩陣
    print('PCA混淆矩陣:')
    print(pca_result)
    print('PCA報告:')
    print(pca_report)
    # 將降為後的資料進行的逆轉換成原本資料
    # data_inverse = myPCA.inverse_transform(pca_test_img)
    # show_img(1,20,data_inverse)

    # 因為LDA模型在SKlearn 維度只接受到0~min(n_features, n_classes-1)之間的維度，因為我們只有40個類別，因此最大維度只能到39
    if(i<40):
        myLDA = LDA(n_components=i).fit(train_img, train_true)
        lda_train_img = myLDA.transform(train_img)
        lda_test_img = myLDA.transform(test_img)

        lda_pred = lda_clf.fit(lda_train_img, train_true)
        lda_test_pred = lda_pred.predict(lda_test_img)
        lda_result = confusion_matrix(test_true, lda_test_pred)

        # 準確率
        lda_accuracy = str(accuracy_score(lda_test_pred, test_true) * 100) + "%"
        lda_report = classification_report(test_true,lda_test_pred)
        # 計算正確預測個數
        true_count = 0
        for j in range(len(lda_result)):
            true_count += lda_result[j][j]
        # 錯誤預測個數
        false_count = 200 - true_count
        # 統計資料
        print('LDA正確預測個數:{}  錯誤預測個數:{} 準確率:{}'.format(true_count, false_count, lda_accuracy))
        # 列印混合矩陣
        print('LDA混淆矩陣:')
        print(lda_result)
        print('LDA報告:')
        print(lda_report)
