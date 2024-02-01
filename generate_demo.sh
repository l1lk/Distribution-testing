python AttackSet.py -seeds ./seeds/mnist/seed_v1/class_0_seed.npy \
-gt ./seeds/mnist/seed_v1/class_0_labels.npy \
-target ./seeds/mnist/seed_v1/class_1_seed.npy \
-target_class 1 \
-o GA_output/GA_100_logits_mnist_lenet5/100_50/seed_0_target_1 \
-pop_num 100 \
-subtotal 50 \
-type mnist \
-model models/mnist_model_0.990.pkl \
-max_iteration 30
