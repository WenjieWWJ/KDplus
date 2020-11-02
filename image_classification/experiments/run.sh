nohup python3 -u no_teacher_pruning.py -d cifar10 -m resnet26 -e 100 -s 0 -g 4 -p_prune 0.4  > ./exp_logs/no_teacher_pruning_resnet26_p_prune_0.4.log 2>&1 &
