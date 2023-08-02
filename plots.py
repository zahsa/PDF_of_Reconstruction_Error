from matplotlib import pyplot as plt

def plot_history(history_loaded300,history_loaded100,history_loaded20):
    plt.plot(history_loaded300["loss"], label="Training, segment size = 300", color='blue')
    plt.plot(history_loaded300["val_loss"], label="Validation, segment size = 300", color='cyan')
    plt.plot(history_loaded100["loss"], label="Training, segmen size = 100", color='orange')
    plt.plot(history_loaded100["val_loss"], label="Validation, segment size = 100", color='gold')
    plt.plot(history_loaded20["loss"], label="Training, segment size = 20", color='magenta')
    plt.plot(history_loaded20["val_loss"], label="Validation, segment size = 20", color='violet')
    # plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    # plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('../Figs/' + 'loss_compare' + '.png')
    plt.show()

savename = 'model1_sog_all_data_split2_fix_sq100'
model100 = keras.models.load_model('../model/' + 'model_' + savename + '.h5')

# load history
savename = 'model1_sog_all_data_split2_fix_sq300'
hist_saved_name = '../model/' + savename + '_hist.npy'
history_loaded300 = np.load(hist_saved_name,allow_pickle='TRUE').item()

savename = 'model1_sog_all_data_split2_fix_sq100'
hist_saved_name = '../model/' + savename + '_hist.npy'
history_loaded100 = np.load(hist_saved_name,allow_pickle='TRUE').item()

savename = 'model1_sog_all_data_split2_fix_sq20'
hist_saved_name = '../model/' + savename + '_hist.npy'
history_loaded20 = np.load(hist_saved_name,allow_pickle='TRUE').item()


plot_history(history_loaded300,history_loaded100,history_loaded20)


# plot evaluation results

# bar chart
x = np.arange(0,8,2) ; width = 0.2
patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

plt.bar(x-0.8, acc_mn_G300, width, color='deepskyblue' , label = 'Accuracy, segment size = 300')
plt.bar(x-0.6, acc_mn_G100, width, color='steelblue' , label = 'Accuracy, segment size = 100')
plt.bar(x-0.4, acc_mn_G20, width, color='mediumblue' , label = 'Accuracy, segment size = 20')

plt.bar(x-0.2,spec_mn_G300 , width, color='violet', label = 'Sensitivity, segment size = 300')
plt.bar(x,spec_mn_G100 , width, color='m' , label = 'Sensitivity, segment size = 100')
plt.bar(x+0.2,spec_mn_G20 , width, color='purple', label = 'Sensitivity, segment size = 20')

plt.bar(x+0.4, sens_mn_G300, width, color='yellow', label = 'Specificity, segment size = 300')
plt.bar(x+0.6, sens_mn_G100, width, color='gold',  label = 'Specificity, segment size = 100')
plt.bar(x+0.8, sens_mn_G20, width, color='goldenrod' ,  label = 'Specificity, segment size = 20')

plt.xticks(x, ['98% CL', '90% CL', '80% CL', '70% CL'])
plt.xlabel("Gaussian Kernel")
plt.ylabel("Scores")
plt.legend(loc='upper right' , bbox_to_anchor=(1.55, 1))
# plt.legend(["Accuracy", "Specificity", "Sensitivity"])
# plt.show()
plt.savefig('../Figs/assesment1_bar_G_pp_ss.png',bbox_inches='tight')

# precision-recall
fig, ax = plt.subplots()
plt.subplots_adjust(hspace=0.5)
colors = ['purple','orange','mediumblue','violet','magenta','chocolate']
# for i in range(4):
#     ax = plt.subplot(1, 4, i+1)
#     recalls = [rec_mn_G300[i], rec_mn_G100[i], rec_mn_G20[i]]
#     precisions = [prec_mn_G300[i], prec_mn_G100[i] , prec_mn_G20[i]]
plt.plot(rec_mn_G300, prec_mn_G300, color=colors[0] , label = 'segment size = 300')
plt.plot(rec_mn_G100, prec_mn_G100, color=colors[4], label = 'segment size = 100')
plt.plot(rec_mn_G20, prec_mn_G20, color=colors[5], label = 'segment size = 20')
ax.legend(loc='lower left')
lines = ax.get_lines()

plt.xlabel('Recall')
plt.ylabel('Precision')

ax2 = ax.twinx()
ax2.scatter(rec_mn_G300[0],prec_mn_G300[0],marker='*',c='green')
plt.scatter(rec_mn_G300[1],prec_mn_G300[1],marker='s',c='green')
plt.scatter(rec_mn_G300[2],prec_mn_G300[2],marker='v',c='green')
plt.scatter(rec_mn_G300[3],prec_mn_G300[3],marker='d',c='green')

ax2.scatter(rec_mn_G100[0],prec_mn_G100[0],marker='*',c='green')
ax2.scatter(rec_mn_G20[0],prec_mn_G20[0],marker='*',c='green')

plt.scatter(rec_mn_G300[1],prec_mn_G300[1],marker='s',c='green')
plt.scatter(rec_mn_G100[1],prec_mn_G100[1],marker='s',c='green')
plt.scatter(rec_mn_G20[1],prec_mn_G20[1],marker='s',c='green')

plt.scatter(rec_mn_G100[2],prec_mn_G100[2],marker='v',c='green')
plt.scatter(rec_mn_G20[2],prec_mn_G20[2],marker='v',c='green')

plt.scatter(rec_mn_G100[3],prec_mn_G100[3],marker='d',c='green')
plt.scatter(rec_mn_G20[3],prec_mn_G20[3],marker='d',c='green')
lines2 = ax2.get_lines()
leg2 = ax2.legend(['98% CL', '90% CL', '80% CL', '70% CL'])
# plt.gca().add_artist(leg2)
xtickslocs = ax.get_xticks()
# x_lbl_loc = [xtickslocs[1],xtickslocs[3],xtickslocs[5],xtickslocs[7]]
# plt.xticks(x_lbl_loc, ['98% CL', '90% CL', '80% CL', '70% CL'])
plt.savefig('../Figs/precision_recall_pp.png')