import argparse
import os
import random
import numpy as np
import tensorflow as tf
import time
from satimg_dataset_processor.data_generator_tf import FireDataGenerator
seed=42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
import platform
import tensorflow_addons as tfa
from tensorflow.python.keras.callbacks import ModelCheckpoint
from wandb.integration.keras import WandbCallback

import wandb
from temporal_models.gru.gru_model import GRUModel
from temporal_models.lstm.lstm_model import LSTMModel
from temporal_models.tcn.tcn import compiled_tcn
from temporal_models.vit_keras import vit

root_path = '/home/z/h/zhao2/TS-SatFire/dataset/'
save_path = '/home/z/h/zhao2/TS-SatFire/checkpoints/'

MAX_EPOCHS = 50

def get_dateset(mode, ts_length, interval, nchannel, batch_size):
    x_dataset = np.load(os.path.join(root_path, 'dataset_train/dataset_train'+mode+'_train_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy'), mmap_mode='r').transpose((0, 3, 4, 2, 1))
    x_dataset = x_dataset.reshape(x_dataset.shape[0]* x_dataset.shape[1]*x_dataset.shape[2], x_dataset.shape[3], x_dataset.shape[4])
    y_dataset = np.load(os.path.join(root_path, 'dataset_train/dataset_train'+mode+'_train_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy'), mmap_mode='r').transpose((0, 3, 4, 2, 1))
    y_dataset = y_dataset.reshape(y_dataset.shape[0]* y_dataset.shape[1]*y_dataset.shape[2], y_dataset.shape[3], y_dataset.shape[4])
    y_af = np.zeros((y_dataset.shape[0],y_dataset.shape[1],2))
    y_af[: ,:, 0] = y_dataset[:, :, 2] == 0
    y_af[:, :, 1] = y_dataset[:, :, 2] > 0
    

    x_dataset_val = np.load(os.path.join(root_path, 'dataset_val/'+mode+'_val_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy'), mmap_mode='r').transpose((0, 3, 4, 2, 1))
    x_dataset_val = x_dataset_val.reshape(x_dataset_val.shape[0]* x_dataset_val.shape[1]*x_dataset_val.shape[2], x_dataset_val.shape[3], x_dataset_val.shape[4])
    y_dataset_val = np.load(os.path.join(root_path, 'dataset_val/'+mode+'_val_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy'), mmap_mode='r').transpose((0, 3, 4, 2, 1))
    y_dataset_val = y_dataset_val.reshape(y_dataset_val.shape[0]* y_dataset_val.shape[1]*y_dataset_val.shape[2], y_dataset_val.shape[3], y_dataset_val.shape[4])    
    y_af_val = np.zeros((x_dataset_val.shape[0],x_dataset_val.shape[1],2))
    y_af_val[: ,:, 0] = y_dataset_val[:, :, 2] == 0
    y_af_val[:, :, 1] = y_dataset_val[:, :, 2] > 0

    x_train, x_val, y_train, y_val = x_dataset, x_dataset_val, y_af, y_af_val
    
    mean = [18.76488,27.441864,20.584806,305.99478,294.31738,14.625097,276.4207,275.16766]
    std = [15.911591,14.879259,10.832616,21.761852,24.703484,9.878246,40.64329,40.7657]
    for i in range(nchannel):
        x_train[..., i] = (x_train[..., i]-mean[i])/std[i]
        x_val[..., i] = (x_val[..., i]-mean[i])/std[i]

    def make_generator(inputs, labels):
        def _generator():
            for input, label in zip(inputs, labels):
                yield input, label

        return _generator
    train_dataset = tf.data.Dataset.from_generator(make_generator(x_train, y_train),
                                                   (tf.float32, tf.int16))
    val_dataset = tf.data.Dataset.from_generator(make_generator(x_val, y_val),
                                                 (tf.float32, tf.int16))
    train_dataset = train_dataset.shuffle(batch_size*10).repeat(MAX_EPOCHS).batch(batch_size)
    val_dataset = val_dataset.prefetch(batch_size*10).repeat(MAX_EPOCHS).batch(batch_size)
    steps_per_epoch = x_train.shape[0]//batch_size
    validation_steps = x_val.shape[0]//(256*256)
    return train_dataset, val_dataset, steps_per_epoch, validation_steps

def wandb_config(model_name, run, num_heads, num_layers, mlp_dim, hidden_size):
    wandb.login()
    wandb.init(project="afba_"+model_name+"_grid_search", entity="zhaoyutim")
    wandb.run.name = 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size) + str(run)
    wandb.config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": MAX_EPOCHS,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_dim": mlp_dim,
        "embed_dim": hidden_size
    }

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-mode', type=str, help='Active Fire or Burned Area Mapping')
    parser.add_argument('-ts', type=int, help='Time-series Length')
    parser.add_argument('-it', type=int, help='Interval between each time-serie')
    parser.add_argument('-nc', type=int, help='Number of Channels')

    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-r', type=int, help='run')
    parser.add_argument('-lr', type=float, help='learning rate')

    parser.add_argument('-nh', type=int, help='number-of-head')
    parser.add_argument('-md', type=int, help='mlp-dimension')
    parser.add_argument('-ed', type=int, help='embedding dimension')
    parser.add_argument('-nl', type=int, help='num_layers')

    parser.add_argument('-test', dest='binary_flag', action='store_true', help='embedding dimension')
    args = parser.parse_args()
    model_name = args.m
    mode = args.mode
    nchannel = args.nc
    interval = args.it
    ts_length = args.ts
    batch_size=args.b
    num_heads=args.nh
    mlp_dim=args.md
    num_layers=args.nl
    hidden_size=args.ed
    is_masked=True

    run = args.r
    lr = args.lr
    learning_rate = lr
    weight_decay = lr / 10
    num_classes=2
    train = args.binary_flag

    input_shape=(ts_length, nchannel)
    data_gen_train = FireDataGenerator(mode, train_test='train', ts_length=ts_length, interval=interval, batch_size=batch_size, input_shape=input_shape, n_channels=nchannel, n_classes=num_classes)
    data_gen_val = FireDataGenerator(mode, train_test='val', ts_length=ts_length, interval=interval, batch_size=batch_size, input_shape=input_shape, n_channels=nchannel, n_classes=num_classes)
    steps_per_epoch = len(data_gen_train)
    validation_steps = len(data_gen_val)

    wandb_config(model_name, run, num_heads, mlp_dim, num_layers, hidden_size)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if model_name=='vit_tiny_custom':
            model = vit.vit_tiny_custom(
                input_shape=input_shape,
                classes=num_classes,
                activation='sigmoid',
                include_top=True,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                hidden_size=hidden_size,
                is_masked=is_masked
            )
        elif model_name == 'gru_custom':
            gru = GRUModel(input_shape, num_classes)
            model = gru.get_model_custom(input_shape, num_classes, num_layers, hidden_size)
        elif model_name == 'lstm_custom':
            lstm = LSTMModel(input_shape, num_classes)
            model = lstm.get_model_custom(input_shape, num_classes, num_layers, hidden_size)
        elif model_name=='tcn':
            model = compiled_tcn(return_sequences=True,
                                 num_feat=input_shape[-1],
                                 num_classes=num_classes,
                                 nb_filters=mlp_dim,
                                 kernel_size=hidden_size,
                                 dilations=[2 ** i for i in range(9)],
                                 nb_stacks=num_layers,
                                 max_len=input_shape[0],
                                 use_weight_norm=True,
                                 use_skip_connections=True)
        else:
            raise('no suport model')

        model.summary()
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        if model_name == 'vit_tiny_custom':
            checkpoint = ModelCheckpoint(os.path.join(save_path, 'afba_'+model_name + '_nopretrained'+'_run'+str(run)+'_'+str(num_heads)+'_'+str(mlp_dim)+'_'+str(hidden_size)+'_'+str(num_layers)+'_'+str(batch_size)+'way1'), monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        elif model_name == 'gru_custom' or model_name == 'lstm_custom':
            checkpoint = ModelCheckpoint(os.path.join(save_path, 'afba_'+model_name+ '_nopretrained'+'_run'+str(run)+'_'+str(hidden_size)+'_'+str(num_layers)), monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        else:
            checkpoint = ModelCheckpoint(os.path.join(save_path, 'afba_'+model_name + '_nopretrained'+'_run'+str(run)), monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        
        loss_fn = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False)
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy_val")
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                
            ],
        )
    if not train:
        model.load_weights(os.path.join('saved_models', 'afba_'+model_name+'mode' + str(mode) + '_nopretrained'+'_run'+str(run)+'_'+str(num_heads)+'_'+str(mlp_dim)+'_'+str(hidden_size)+'_'+str(num_layers)+'_'+str(batch_size)+'_'+str(is_masked)+'way2'))
    else:
        for epoch in range(MAX_EPOCHS):
            start_time = time.time()
            print("\nStart of epoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(data_gen_train):
                b, h, w, t, c = x_batch_train.shape[0], x_batch_train.shape[1], x_batch_train.shape[2], x_batch_train.shape[3], x_batch_train.shape[4]
                x_batch_train = tf.reshape(x_batch_train, (b*h*w, t, c))
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)
                    logits = tf.reshape(logits, (b, h, w, t, c))
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_acc_metric.update_state(y_batch_train, logits)

                if step % 10 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))
                
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            train_acc_metric.reset_states()

            for x_batch_val, y_batch_val in data_gen_val:
                b, h, w, t, c = x_batch_val.shape[0], x_batch_val.shape[1], x_batch_val.shape[2], x_batch_val.shape[3], x_batch_val.shape[4]
                val_logits = model(x_batch_val, training=False)
                val_logits = tf.reshape(val_logits, (b, h, w, t, c))
                val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))