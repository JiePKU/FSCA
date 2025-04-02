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

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    split_N = 0

    for batch_idx, ((member_image_inputs,  member_text_inputs, member_sim_score), (non_member_image_inputs, non_member_text_inputs, non_member_sim_score)) in private_testset:

        member_image_inputs = member_image_inputs.to(device) 
        non_member_image_inputs = non_member_image_inputs.to(device) 
    
        member_text_inputs = member_text_inputs.to(device) 
        non_member_text_inputs = non_member_text_inputs.to(device) 
        
        # print(member_sim_score.reshape(6,1))

        with torch.no_grad():

            member_features = obtain_membership_feature(member_image_inputs, member_text_inputs, args.feature_type)
            non_member_features = obtain_membership_feature(non_member_image_inputs, non_member_text_inputs, args.feature_type)
            
            if split_N>0:
                member_features = torch.cat((member_features, non_member_features[:split_N]))
                non_member_features = non_member_features[split_N:]

                v_is_member_labels = torch.from_numpy(
                np.reshape(np.concatenate((np.ones(member_features.size(0)-split_N), np.zeros(split_N))), [-1, 1])).to(device).float()
            
            else:    
                v_is_member_labels = torch.from_numpy(
                np.reshape(np.ones(member_features.size(0)), [-1, 1])).to(device).float()

            v_is_non_member_labels = torch.from_numpy(
            np.reshape(np.zeros(non_member_features.size(0)), [-1, 1])).to(device).float()

            # attack_model_input = torch.cat((member_features, non_member_features))  # torch.cat((member_features))

            r = np.arange(v_is_member_labels.size()[0]).tolist()
            random.shuffle(r)
            member_features = member_features[r]
            v_is_member_labels = v_is_member_labels[r]

            member_output = adversary(member_features)
            non_member_output = adversary(non_member_features)

        ## for member_out

        if (member_output[v_is_member_labels.long()==1] > 0.5).int().sum()>0:
            TP += 1
        else:
            FN += 1
        
        if (non_member_output > 0.5).int().sum()>0:
            FP += 1
        else:
            TN += 1
            
        if batch_idx == 99:
            break
 
    metrics = compute_metrics(TP, FP, TN, FN)

    return metrics["Accuracy"], metrics["Precision"], metrics["Recall"]


def compute_metrics(TP, FP, TN, FN):
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score
    }



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

