import csv
import tensorflow as tf
import matplotlib.pyplot as plt


# path1  pred
# path2  label

def ROC(path1,path2,iter=500):
    pred = []
    label = []
    with open(path1, encoding="utf-8") as f:
        lines = list(csv.reader(f))
        for i in lines:
            temp = []
            temp.append(float(i[0]))
            temp.append(float(i[1]))
            pred.append(temp)

    with open(path2, encoding="utf-8") as f:
        lines = list(csv.reader(f))
        for i in lines:
            temp = []
            temp.append(float(i[0]))
            temp.append(float(i[1]))
            label.append(temp)

    Num_pos = label.count([1.0, 0.0])
    Num_neg = label.count([0.0, 1.0])
    print("num_neg:",Num_neg,'  num_pos:',Num_pos)
    
    chazhi = []
    for i in pred:
        temp = i[0]-i[1]
        chazhi.append(temp)

    max1 = float(max(chazhi))
    min1 = float(min(chazhi))
    TPR = []
    FPR = []

    for i in range(iter+1):
        bool_ = min1 + i*(max1-min1)/iter
        tp = 0
        tn = 0
        k = 0
        for j in chazhi:
            if j>bool_ and label[k] == [1.0,0.0]:
                tp = tp + 1
                k =k + 1
            elif j<=bool_ and label[k] == [0.0,1.0]:
                tn = tn + 1
                k = k + 1
            else:
                k = k + 1
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
    # plt.axis([0.0, 1.0, 0.0, 1.0])
    # plt.title('ROC curve on independent test set')
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.grid(color="k", linestyle=":")
    # plt.plot(FPR, TPR)
    # plt.show()

    return area