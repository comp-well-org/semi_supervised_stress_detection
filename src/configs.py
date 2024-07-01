dataset = "SMILE"
training_mode = "supervised"

noise_sigma = 0.05
warp_sigma = 0.05
load_pretrained = True

BATCH_SIZE=32
BATCH_SIZE_UNLABELED=64
LR = 1e-3
LR_decay_step = 15
epochs = 50
supResolution = 15

save_path = "/home/hy29/rdf/semi_supervised_v2/results/supervised/{}mins_lr{}/".format(supResolution, LR)
save_path_log = "./logs"
save_model_path_gsr = "/home/hy29/rdf/semi_supervised_v2/results/representation_learning/SMILE_GSR/supervised_1_checkpoint_200.pth"
save_model_path_ecg = "/home/hy29/rdf/semi_supervised_v2/results/representation_learning/SMILE_ECG/supervised_1_checkpoint_200.pth"
tb_comments = " "