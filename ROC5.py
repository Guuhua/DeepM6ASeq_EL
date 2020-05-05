import csv

import matplotlib.pyplot as plt


# path1  pred
# path2  label

def ROC(path1,iter=500, low=0.55,high=0.70):
    pred1 = []
    label1 = []
    pred2 = []
    pred3 = []
    pred4 = []
    pred5 = []
    
    with open(path1+'0AUROC/predLSTMCNN0.csv', encoding="utf-8") as f:
        lines = list(csv.reader(f))
        for i in lines:
            temp = []
            temp.append(float(i[0]))
            temp.append(float(i[1]))
            pred1.append(temp)
    with open(path1+'1AUROC/predLSTMCNN1.csv', encoding="utf-8") as f:
        lines = list(csv.reader(f))
        for i in lines:
            temp = []
            temp.append(float(i[0]))
            temp.append(float(i[1]))
            pred2.append(temp)
    with open(path1+'2AUROC/predLSTMCNN2.csv', encoding="utf-8") as f:
        lines = list(csv.reader(f))
        for i in lines:
            temp = []
            temp.append(float(i[0]))
            temp.append(float(i[1]))
            pred3.append(temp)
    with open(path1+'3AUROC/predLSTMCNN3.csv', encoding="utf-8") as f:
        lines = list(csv.reader(f))
        for i in lines:
            temp = []
            temp.append(float(i[0]))
            temp.append(float(i[1]))
            pred4.append(temp)
    with open(path1+'23AUROC/predLSTMCNN23.csv', encoding="utf-8") as f:
        lines = list(csv.reader(f))
        for i in lines:
            temp = []
            temp.append(float(i[0]))
            temp.append(float(i[1]))
            pred5.append(temp)

    with open(path1+'0AUROC/labelLSTMCNN0.csv', encoding="utf-8") as f:
        lines = list(csv.reader(f))
        for i in lines:
            temp = []
            temp.append(float(i[0]))
            temp.append(float(i[1]))
            label1.append(temp)

    Num_pos = label1.count([1.0, 0.0])
    Num_neg = label1.count([0.0, 1.0])
    print("num_neg:",Num_neg,'  num_pos:',Num_pos)

    chazhi1 = []
    chazhi2 = []
    chazhi3 = []
    chazhi4 = []
    chazhi5 = []

    for i in pred1:
        temp = i[0]-i[1]
        chazhi1.append(temp)

    for i in pred2:
        temp = i[0]-i[1]
        chazhi2.append(temp)

    for i in pred3:
        temp = i[0]-i[1]
        chazhi3.append(temp)

    for i in pred4:
        temp = i[0]-i[1]
        chazhi4.append(temp)

    for i in pred5:
        temp = i[0]-i[1]
        chazhi5.append(temp)

    max1 = float(max(chazhi1))
    min1 = float(min(chazhi1))

    max2 = float(max(chazhi2))
    min2 = float(min(chazhi2))

    max3 = float(max(chazhi3))
    min3 = float(min(chazhi3))

    max4 = float(max(chazhi4))
    min4 = float(min(chazhi4))

    max5 = float(max(chazhi5))
    min5 = float(min(chazhi5))

    max6 = max(max1,max2,max3,max4,max5)
    min6 = min(min1,min2,min3,min4,min5)

    TPR = []
    FPR = []

    for i in range(iter+1):
        if i == 0 or (i/iter > low and i/iter < high) or i==iter:
            bool_ = min6 + i*(max6-min6)/iter
            tp = 0
            tn = 0
            for j in range(len(chazhi1)):
                temp11 = 0
                if chazhi1[j]>bool_:
                    temp11 = temp11 + 1
                else:
                    temp11 = temp11 - 1
                if chazhi2[j]>bool_:
                    temp11 = temp11 + 1
                else:
                    temp11 = temp11 - 1
                if chazhi3[j]>bool_:
                    temp11 = temp11 + 1
                else:
                    temp11 = temp11 - 1
                if chazhi4[j]>bool_:
                    temp11 = temp11 + 1
                else:
                    temp11 = temp11 - 1    
                if chazhi5[j]>bool_:
                    temp11 = temp11 + 1
                else:
                    temp11 = temp11 - 1

                if temp11>0 and label1[j]==[1.0,0.0]:
                    tp = tp + 1
                elif temp11<0 and label1[j]==[0.0,1.0]:
                    tn = tn + 1

            print("iter: ", i, " TP: ", tp," TN: ", tn)
            tpr_ = tp/Num_pos
            fpr_ = 1 - tn/Num_neg
            TPR.append(tpr_)
            FPR.append(fpr_)
    
    # 计算面积
    area = 0
    for i in range(len(TPR)-1):
        temp = (-FPR[i+1]+FPR[i])*(TPR[i]+TPR[i+1])/2
        area = area + temp
    print('AUROC: ', area)

    # 画出曲线
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.title('ROC curve on independent test set')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid(color="k", linestyle=":")
    plt.plot(FPR, TPR)
    plt.show()

    return area,FPR,TPR


if __name__ == '__main__':

    path_ = "/home/guhua/Documents/BIO/model_data/maturemRNA/seq/"
    type_ = ["A549","CD8T","HEK293_abacm","HEK293_sysy","HeLa","MOLM13"]

    step = 2000

    tmp = path_ + type_[0] + '/'
    print(tmp)
    AU0, _, _ = ROC(tmp, iter=step, low=0.17, high=0.26)

    tmp = path_ + type_[1] + '/'
    print(tmp)
    AU1, _, _ = ROC(tmp, iter=step, low=0.16,high=0.28)

    tmp = path_ + type_[2] + '/'
    print(tmp)
    AU2, _, _ = ROC(tmp, iter=step, low=0.14,high=0.30)

    tmp = path_ + type_[3] + '/'
    print(tmp)
    AU3, _, _ = ROC(tmp, iter=step, low=0.16, high=0.21)

    tmp = path_ + type_[4] + '/'
    print(tmp)
    AU4, _, _ = ROC(tmp, iter=step, low=0.21,high=0.32)

    tmp = path_ + type_[5] + '/'
    print(tmp)
    AU5, _, _ = ROC(tmp, iter=step, low=0.20,high=0.32)

    aver_AU = (AU0+AU1+AU2+AU3+AU4+AU5)/6
    print(aver_AU)

    # path_ = "/home/guhua/Documents/BIO/model_data/maturemRNA/onehot/"
    # type_ = ["A549","CD8T","HEK293_abacm","HEK293_sysy","HeLa","MOLM13"]

    # step = 2000

    # tmp = path_ + type_[0] + '/'
    # print(tmp)
    # AU0, _, _ = ROC(tmp, iter=step, low=0.14, high=0.22)

    # tmp = path_ + type_[1] + '/'
    # print(tmp)
    # AU1, _, _ = ROC(tmp, iter=step, low=0.12,high=0.18)

    # tmp = path_ + type_[2] + '/'
    # print(tmp)
    # AU2, _, _ = ROC(tmp, iter=step, low=0.07,high=0.14)

    # tmp = path_ + type_[3] + '/'
    # print(tmp)
    # AU3, _, _ = ROC(tmp, iter=step, low=0.07, high=0.16)

    # tmp = path_ + type_[4] + '/'
    # print(tmp)
    # AU4, _, _ = ROC(tmp, iter=step, low=0.18,high=0.30)

    # tmp = path_ + type_[5] + '/'
    # print(tmp)
    # AU5, _, _ = ROC(tmp, iter=step, low=0.18,high=0.28)

    # aver_AU = (AU0+AU1+AU2+AU3+AU4+AU5)/6
    # print(aver_AU)

    '''
    path_ = "/home/guhua/Documents/BIO/model_data/maturemRNA/embedding/"
    type_ = ["A549","CD8T","HEK293_abacm","HEK293_sysy","HeLa","MOLM13"]

    step = 2000

    tmp = path_ + type_[0] + '/'
    print(tmp)
    AU0, _, _ = ROC(tmp, iter=step, low=0.16, high=0.25)

    tmp = path_ + type_[1] + '/'
    print(tmp)
    AU1, _, _ = ROC(tmp, iter=step, low=0.13,high=0.20)

    tmp = path_ + type_[2] + '/'
    print(tmp)
    AU2, _, _ = ROC(tmp, iter=step, low=0.08,high=0.14)

    tmp = path_ + type_[3] + '/'
    print(tmp)
    AU3, _, _ = ROC(tmp, iter=step, low=0.14, high=0.21)

    tmp = path_ + type_[4] + '/'
    print(tmp)
    AU4, _, _ = ROC(tmp, iter=step, low=0.20,high=0.30)

    tmp = path_ + type_[5] + '/'
    print(tmp)
    AU5, _, _ = ROC(tmp, iter=step, low=0.16,high=0.27)

    aver_AU = (AU0+AU1+AU2+AU3+AU4+AU5)/6
    print(aver_AU)
    '''

    '''
    # path_ = "/home/guhua/Documents/BIO/model_data/FULLtranscript/onehot/"
    # type_ = ["A549","CD8T","HEK293_abacm","HEK293_sysy","HeLa","MOLM13"]

    # step = 2000

    # tmp = path_ + type_[0] + '/'
    # print(tmp)
    # AU0, _, _ = ROC(tmp, iter=step, low=0.26, high=0.37)

    # tmp = path_ + type_[1] + '/'
    # print(tmp)
    # AU1, _, _ = ROC(tmp, iter=step, low=0.22,high=0.30)

    # tmp = path_ + type_[2] + '/'
    # print(tmp)
    # AU2, _, _ = ROC(tmp, iter=step, low=0.22,high=0.30)

    # tmp = path_ + type_[3] + '/'
    # print(tmp)
    # AU3, _, _ = ROC(tmp, iter=step, low=0.23, high=0.30)

    # tmp = path_ + type_[4] + '/'
    # print(tmp)
    # AU4, _, _ = ROC(tmp, iter=step, low=0.32,high=0.45)

    # tmp = path_ + type_[5] + '/'
    # print(tmp)
    # AU5, _, _ = ROC(tmp, iter=step, low=0.20,high=0.33)

    # aver_AU = (AU0+AU1+AU2+AU3+AU4+AU5)/6
    # print(aver_AU)
    '''

    '''
    path_ = "/home/guhua/Documents/BIO/model_data/FULLtranscript/embedding/"
    type_ = ["A549","CD8T","HEK293_abacm","HEK293_sysy","HeLa","MOLM13"]

    step = 2000

    tmp = path_ + type_[0] + '/'
    print(tmp)
    AU0, _, _ = ROC(tmp, iter=step, low=0.27, high=0.37)

    tmp = path_ + type_[1] + '/'
    print(tmp)
    AU1, _, _ = ROC(tmp, iter=step, low=0.26,high=0.36)

    tmp = path_ + type_[2] + '/'
    print(tmp)
    AU2, _, _ = ROC(tmp, iter=step, low=0.23,high=0.30)

    tmp = path_ + type_[3] + '/'
    print(tmp)
    AU3, _, _ = ROC(tmp, iter=step, low=0.19, high=0.28)

    tmp = path_ + type_[4] + '/'
    print(tmp)
    AU4, _, _ = ROC(tmp, iter=step, low=0.32,high=0.48)

    tmp = path_ + type_[5] + '/'
    print(tmp)
    AU5, _, _ = ROC(tmp, iter=step, low=0.28,high=0.39)

    aver_AU = (AU0+AU1+AU2+AU3+AU4+AU5)/6
    print(aver_AU)
    '''