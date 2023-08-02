
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KernelDensity

def difference_method(x_input):
    x_transformed = x_input.diff().dropna()
    return x_transformed


def standardization(x_input):
    training_min = x_input.min()
    training_max = x_input.max()
    x_standardized = (x_input - training_min) / (training_max - training_min)
#     print("Number of training samples:", len(x_standardized))
    return x_standardized

# Generated training sequences for use in the model.
def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
#         print(i)
    return np.stack(output)
        
def test_results(model,n_test,test_ind,dataset,pdf_estimator,time_steps, label_thresh):

    test_acc = np.zeros((4,n_test))
    test_sens = np.zeros((4,n_test))
    test_spe = np.zeros((4,n_test))
    test_cm = np.zeros((4,n_test,2,2))
    test_prec = np.zeros((4,n_test))
    test_rec = np.zeros((4,n_test))
    test_fscore = np.zeros((4,n_test))
    y_dict = {0:{} , 1:{} , 2:{} , 3:{}}
    y_pred_dict = {0:{} , 1:{} , 2:{} , 3:{}}
    y_anom_dict = {0:{} , 1:{} , 2:{} , 3:{}}
    y_norm_dict = {0:{} , 1:{} , 2:{} , 3:{}}
    test_num = -1
    for i , ti in enumerate(test_ind):
        print('test data' , i, 'ind', ti)
#         print('----------------------------------------------------')
        df = dataset[dataset['trajectory']==ti]
        df_sog = df.loc[:,'sog']
        if df_sog.shape[0] > 10000 and ~df_sog.isnull().values.any(): 
    #         print(df_sog.shape)
            df_sog = pd.DataFrame(df_sog)
            sog_diff = difference_method(df_sog)
            sog_diff_stand = standardization(sog_diff)
#             print(sog_diff_stand.shape)
            x_test_diff_stand_seq = create_sequences(sog_diff_stand.values, time_steps)

            if np.shape(np.argwhere(np.isnan(x_test_diff_stand_seq)))[0] > 0:
                print('nan-----------------')
                continue
            test_num +=1
            if test_num==0:
                xx_test = x_test_diff_stand_seq
    #         print('shape',x_test_diff_stand_seq.shape)
            if test_num > 0:
                xx_test = np.concatenate((xx_test,x_test_diff_stand_seq),axis=0)


            x_test_pred = model.predict(x_test_diff_stand_seq)

            test_mae_loss = np.mean(np.abs(x_test_pred - x_test_diff_stand_seq), axis=1)

            confidence_thresh_list, pdf, cdf = confidenceLevelThresh(test_mae_loss,kernel_name = pdf_estimator)

#             print('CL',confidence_thresh_list)
            for ci,cl in enumerate(confidence_thresh_list):
                accuracy , sensitivity, specificity, cm, precision, recall, fscore, y, y_pred , y_anom, y_norm = evaluate(test_mae_loss, time_steps,sog_diff_stand, cl, label_thresh)
                test_acc[ci][test_num] = accuracy
                test_sens[ci][test_num] = sensitivity
                test_spe[ci][test_num] = specificity
    #             print('cm',cm.shape)
                test_cm[ci][test_num,:] = cm
                test_prec[ci][test_num] = precision
                test_rec[ci][test_num] = recall
                test_fscore[ci][test_num] = fscore    
                y_dict[ci][test_num] = y
                y_pred_dict[ci][test_num] = y_pred
                y_anom_dict[ci][test_num] = y_anom
                y_norm_dict[ci][test_num] = y_norm


    return xx_test, test_acc, test_sens, test_spe , test_cm, test_prec , test_rec, test_fscore, test_num, y_dict , y_pred_dict , y_anom_dict , y_norm_dict 

def confidenceLevelThresh(train_mae_loss,kernel_name):
    x = train_mae_loss
    xsorted = np.sort(x,axis=0)

    if kernel_name == 'histogram':
            (counts, edges, patches) = plt.hist(train_mae_loss, bins=25)
            pdf = counts / sum(counts)
            cdf = np.cumsum(pdf)
    else:
        kde = KernelDensity(bandwidth=np.max(x)/100, kernel= kernel_name )
        kde.fit(x)

        logprob_epanech_error = kde.score_samples(xsorted)

        kde_pdf = np.exp(logprob_epanech_error)

        pdf = kde_pdf/len(kde_pdf)
        cdf = np.cumsum(pdf)/sum(pdf)

#         kde_vals = kde_pdf.tolist()

    q_kde_98 = np.array(np.percentile(cdf, 98, axis=0))
    pos = (np.abs(cdf-q_kde_98)).argmin()
    confidence_thresh_98 = xsorted[pos]

    q_kde_90 = np.array(np.percentile(cdf, 90, axis=0))
    pos = (np.abs(cdf-q_kde_90)).argmin()
    confidence_thresh_90 = xsorted[pos]

    q_kde_80 = np.array(np.percentile(cdf, 80, axis=0))
    pos = (np.abs(cdf-q_kde_80)).argmin()
    confidence_thresh_80 = xsorted[pos]

    q_kde_70 = np.array(np.percentile(cdf, 70, axis=0))
    pos = (np.abs(cdf-q_kde_70)).argmin()
    confidence_thresh_70 = xsorted[pos]

    return [confidence_thresh_98, confidence_thresh_90,confidence_thresh_80, confidence_thresh_70] , pdf, cdf

def evaluate(test_mae_loss, time_step , sog_diff_stand, confidence_thresh,label_thresh):
    mean_ground = np.mean(sog_diff_stand)
    diff_mean = np.abs(sog_diff_stand - mean_ground) 
    ind_traj = sog_diff_stand


    true_binary_labels = diff_mean > label_thresh

    true_binary_labels = true_binary_labels.astype(int)    
    true_binary_labels = true_binary_labels[0:len(true_binary_labels)-time_step+1]

#     print('true_binary_labels shape' , true_binary_labels.shape)

    true_anom_ind = np.where(true_binary_labels)
    true_norm_ind = np.where(true_binary_labels == 0)

#     print('true_anom_ind',len(true_anom_ind[0]))
#     print('true_norm_ind',len(true_norm_ind[0]))

    true_anom_num = np.sum(true_binary_labels)
    true_norm_num = true_binary_labels.shape[0] - true_anom_num


    anomalies = test_mae_loss > confidence_thresh
    pred_anom_ind = np.where(anomalies)
    pred_norm_ind = np.where(test_mae_loss <= confidence_thresh)

#     print('pred_anom_ind',len(pred_anom_ind[0]))      
#     print('pred_norm_ind',len(pred_norm_ind[0]))      


    pred_binary_labels = anomalies.astype(int)




    correct_pred_anom_num = len(set(list(true_anom_ind)[0]).intersection(set(list(pred_anom_ind)[0])))

    correct_pred_norm_num = len(set(list(true_norm_ind)[0]).intersection(set(list(pred_norm_ind)[0])))

    accuracy = (correct_pred_anom_num + correct_pred_norm_num) / (true_anom_num + true_norm_num)

    sensitivity = correct_pred_anom_num / true_anom_num #TP/TP+FN
    specificity = correct_pred_norm_num / true_norm_num #TN/FP+TN

#     print('num of correct detected anom', correct_pred_anom_num)
#     print('num of correct detected norm', correct_pred_norm_num)


#     print('accuracy', accuracy[0])
#     print('sensitivity', sensitivity[0])
#     print('specificity', specificity[0])

    y_true = true_binary_labels
    y_true = y_true.values

    y_pred = pred_binary_labels


#     cm = confusion_matrix(y_true, y_pred, normalize='all')
    cm = confusion_matrix(y_true, y_pred, normalize='true')
#     cm = confusion_matrix(y_true, y_pred, normalize='true')


#     labels = ['anomaly','normal']
#     cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
#     cmd.plot()

    precision = precision_score(y_true, y_pred, average='weighted')
#     print('Precision: %.3f' % precision)

    recall = recall_score(y_true, y_pred, average='weighted')
#     print('Recall: %.3f' % recall)

    fscore = f1_score(y_true, y_pred, average='weighted')
#     print('F-Measure: %.3f' % fscore)

    return accuracy[0], sensitivity[0], specificity[0], cm, precision, recall, fscore, y_true, y_pred,list(true_anom_ind)[0],list(true_norm_ind)[0]

    