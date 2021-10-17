import torch        
from utils.loss import weighted_class_bceloss
from utils.loss import vae_loss_function
from utils.loss import l1_penalty
from tqdm import tqdm


def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True


def train_mlp(mlp_model, train_dataset, valid_dataset, train_dataloader, valid_dataloader, optimizer, weights, scheduler, joint_train_flag, epoch_num):
    # train the model
    # epoch = args.epoch
    best_valid_loss = 1e10
    best_valid_acc = []

    for epoch in range(epoch_num):
        num_positive, num_negative = 0., 0.
        num_correct_positive, num_correct_negative = 0, 0
        total_loss = 0.

        mlp_model.train()
        for train_user_features, train_user_lens, train_hashtag_features, train_hashtag_lens, train_bow_hashtag_features, labels in tqdm(train_dataloader):
            if torch.cuda.is_available():
                train_user_features = train_user_features.cuda()
                train_user_lens = train_user_lens.cuda()
                train_hashtag_features = train_hashtag_features.cuda()
                train_hashtag_lens = train_hashtag_lens.cuda()
                train_bow_hashtag_features = train_bow_hashtag_features.cuda()
                labels = labels.cuda()


            # train process-----------------------------------
            optimizer.zero_grad()

            
            if (not joint_train_flag):
                fix_model(mlp_model.get_vae_model())
                weight, topic_words, pred_labels, recon_batch, data_bow, mu, logvar = mlp_model(joint_train_flag, train_user_features, train_user_lens, train_hashtag_features, train_hashtag_lens, train_bow_hashtag_features)
                mlp_loss = weighted_class_bceloss(pred_labels, labels.reshape(-1, 1), weights)
                
                loss = 100*mlp_loss
                total_loss += (loss.item() * len(labels))
            else:
                unfix_model(mlp_model.get_vae_model())
                # forward pass
                weight, topic_words, pred_labels, recon_batch, data_bow, mu, logvar = mlp_model(joint_train_flag, train_user_features, train_user_lens, train_hashtag_features, train_hashtag_lens, train_bow_hashtag_features)

                # compute loss
                mlp_loss = weighted_class_bceloss(pred_labels, labels.reshape(-1, 1), weights)
                
                vae_loss = vae_loss_function(recon_batch, data_bow, mu, logvar)
                vae_loss = vae_loss + mlp_model.get_vae_model().l1_strength * l1_penalty(mlp_model.get_vae_model().fcd1.weight)
                loss = 100*mlp_loss + vae_loss/100
                total_loss += (loss.item() * len(labels))
            

            for pred_label, label in zip(pred_labels, labels.reshape(-1, 1)):
                if label == 1:
                    num_positive += 1
                    if pred_label > 0.95:
                        num_correct_positive += 1
                else:
                    num_negative += 1
                    if pred_label < 0.95:
                        num_correct_negative += 1

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=20.0, norm_type=2)
            optimizer.step()
        
        print('train positive_acc: %f    train negative_acc: %f    train_loss: %f' % \
            ((num_correct_positive / num_positive), (num_correct_negative / num_negative), (total_loss / len(train_dataset))))


        num_positive, num_negative = 0., 0.
        num_correct_positive, num_correct_negative = 0, 0
        total_loss = 0.

        # best_model = Mlp(768, 256)
        # best_model.load_state_dict(torch.load())
        # if torch.cuda.is_available():
        #     best_model = best_model.cuda()
        mlp_model.eval()
        with torch.no_grad():
            for user_features, user_lens, hashtag_features, hashtag_lens, bow_hashtag_features, labels in tqdm(valid_dataloader):
                if torch.cuda.is_available():
                    user_features = user_features.cuda()
                    user_lens = user_lens.cuda()
                    hashtag_features = hashtag_features.cuda()
                    hashtag_lens = hashtag_lens.cuda()
                    bow_hashtag_features = bow_hashtag_features.cuda()
                    labels = labels.cuda()
               
                if (not joint_train_flag) :
                    fix_model(mlp_model.get_vae_model())
                    weight, topic_words, pred_labels, recon_batch, data_bow, mu, logvar = mlp_model(joint_train_flag, user_features, user_lens, hashtag_features, hashtag_lens, bow_hashtag_features)
                    mlp_loss = weighted_class_bceloss(pred_labels, labels.reshape(-1, 1), weights)
                    loss = 100*mlp_loss
                    total_loss += (loss.item() * len(labels))
                else:
                    unfix_model(mlp_model.get_vae_model())
                    weight, topic_words, pred_labels, recon_batch, data_bow, mu, logvar = mlp_model(joint_train_flag, user_features, user_lens, hashtag_features, hashtag_lens, bow_hashtag_features)
                    mlp_loss = weighted_class_bceloss(pred_labels, labels.reshape(-1, 1), weights)
                    vae_loss = vae_loss_function(recon_batch, data_bow, mu, logvar)
                    vae_loss = vae_loss + mlp_model.get_vae_model().l1_strength * l1_penalty(mlp_model.get_vae_model().fcd1.weight)
                    loss = 100*mlp_loss + vae_loss/100
                    total_loss += (loss.item() * len(labels))
                
                       
                for pred_label, label in zip(pred_labels, labels.reshape(-1, 1)):
                    if label == 1:
                        num_positive += 1
                        if pred_label > 0.95:
                            num_correct_positive += 1
                    else:
                        num_negative += 1
                        if pred_label < 0.95:
                            num_correct_negative += 1

        print('epoch: %d      valid positive_acc: %f   valid negative_acc: %f     valid_loss: %f' % \
            ((epoch + 1), (num_correct_positive / num_positive), (num_correct_negative / num_negative), (total_loss / len(valid_dataset))))
        scheduler.step(total_loss / len(valid_dataset))
        print('learning rate:  %f' % optimizer.param_groups[0]['lr'])
        best_valid_acc.append(num_correct_positive / num_positive + num_correct_negative / num_negative)

        if total_loss < best_valid_loss:
            best_valid_loss = total_loss
            best_epoch = epoch
            print('Current best!')
            torch.save(mlp_model.state_dict(), './ModelRecords'+f'/best_model_{epoch}.pt')
        torch.save(mlp_model.state_dict(), './ModelRecords'+f'/model_{epoch}.pt')
    
    # choose the best epoch after the epoch 50 best on the acccuracy metric of validation set
    try:
        best_epoch = best_valid_acc.index(max(best_valid_acc[50:])) 
    except:
        best_epoch = 0
    return best_epoch
