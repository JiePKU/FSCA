import torch
import torch.nn.functional as F
# from log import  print_and_log
import numpy as np
import random

from mia_lib.mia_util import obtain_membership_feature
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

def mia_evaluate(args, adversary, device, private_testset, is_test_set=False):
    
    adversary.eval()
    correct = 0
    n = 0
    gain = 0
    binary_confusion_matrix = 0
    all_prabability = []
    all_label = []

    for batch_idx, ((member_image_inputs,  member_text_inputs), (non_member_image_inputs, non_member_text_inputs)) in private_testset:

        member_image_inputs = member_image_inputs.to(device) 
        non_member_image_inputs = non_member_image_inputs.to(device) 
    
        member_text_inputs = member_text_inputs.to(device) 
        non_member_text_inputs = non_member_text_inputs.to(device) 
    
        with torch.no_grad():

            member_features = obtain_membership_feature(member_image_inputs, member_text_inputs, args.feature_type)
            non_member_features = obtain_membership_feature(non_member_image_inputs, non_member_text_inputs, args.feature_type)
            
            v_is_member_labels = torch.from_numpy(
            np.reshape(np.concatenate((np.ones(member_features.size(0)), np.zeros(non_member_features.size(0)))), [-1, 1])).to(device).float()

            attack_model_input = torch.cat((member_features, non_member_features))
            
        r = np.arange(v_is_member_labels.size()[0]).tolist()
        random.shuffle(r)
        attack_model_input = attack_model_input[r]
        v_is_member_labels = v_is_member_labels[r]
        member_output = adversary(attack_model_input)
        
        all_prabability.append(member_output)
        all_label.append(v_is_member_labels)

        correct += ((member_output > 0.5) == v_is_member_labels).sum().item()
        binary_confusion_matrix += confusion_matrix(v_is_member_labels.cpu(), (member_output.cpu() > 0.5)+0)
        n += member_output.size()[0]
        gain += ((v_is_member_labels==1)*(member_output-0.5)).sum() + ((0.5-member_output)*(v_is_member_labels==0)).sum()

    print(binary_confusion_matrix)

    TP, FP, FN = binary_confusion_matrix[1,1], binary_confusion_matrix[0,1], binary_confusion_matrix[1,0] 
    
    print('\n{}: MIA accuracy: {}/{} ({:.3f}%) MIA Gain: {:.3f}% MIA Precision:{}/{} ({:.3f}%) MIA Recall:{}/{} ({:.3f}%) MIA F1: {:.3f}% \n'.format(
        'MIA Test evaluation' if is_test_set else 'MIA Evaluation',
        correct, n, 100. * correct / float(n), 100. *gain/float(n), TP, (TP+FP), 100. *TP/float(TP+FP), TP, (TP+FN), 100. *TP/float(TP+FN), 100.0 * 2* (TP/float(TP+FP)) * (TP/float(TP+FN))/(TP/float(TP+FP) + TP/float(TP+FN)) ))

    all_prabability = torch.cat(all_prabability, dim=0).squeeze(dim=1).cpu().detach().numpy()
    all_label = torch.cat(all_label, dim=0).squeeze(dim=1).cpu().detach().numpy()
    cal_auc_fpr(all_prabability, all_label)

    return correct / float(n)


def cal_auc_fpr(probability, labels):

    best_threshold = None
    best_accuracy = 0.0

    min_threshold = min(probability)
    max_threshold = max(probability)
    threshold_step = (max_threshold - min_threshold) / 2000

    for threshold in list(np.arange(min_threshold, max_threshold, threshold_step)):
        predicted_values = [1 if value > threshold else 0 for value in probability]

        accuracy = accuracy_score(labels, predicted_values)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold


    auc = roc_auc_score(labels, [(e - min_threshold) / (max_threshold - min_threshold) for e in probability])

    fpr, tpr, _ = roc_curve(labels, [(e - min_threshold) / (max_threshold - min_threshold) for e in probability])
    idx_1_percent_fpr = next(i for i, fpr_value in enumerate(fpr) if fpr_value >= 0.01)
    tpr_at_1_percent_fpr = tpr[idx_1_percent_fpr]

    print( '\n |   best_accuracy, best_threshold, th% :', best_accuracy, best_threshold,
          (best_threshold - min_threshold) / (max_threshold - min_threshold), "|    AUC Score:", auc,  "|   tpr_at_1_percent_fpr:", tpr_at_1_percent_fpr, "\n")

