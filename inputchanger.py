import wandb
import argparse
from wandb.keras import WandbCallback
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
#import matplotlib.pyplot as plt
#import seaborn as sns
#from matplotlib.pyplot import figure
#import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape")
    args = parser.parse_args()
    #print(args)
    wandb.login()

    def data_load(input_shape):
        x = np.zeros((4800,11))
        # input_path = "content/numpy_data"
        # interp_files=[]
        # for dir in os.listdir(input_path):
        #     for file in os.listdir(input_path + '/' + dir):
        #         if file.endswith(".npy"):
        #             interp_files.append(input_path + '/' + dir + '/' + file)

        # for file in interp_files:
        #     temp = np.load(file)
        #     # temp = temp.reshape((4800,11,1))
        #     x = np.append(x, temp, axis=0)
        temp = np.load("content/numpy_data/A-B/1.npy")
        x = np.append(x, temp, axis=0)
        x = x[4800:, :]
        x_vals = x[:, :input_shape]
        y_vals = x[:, input_shape:]
        x_vals = x_vals.reshape((4800, input_shape, 1))
        y_vals = y_vals.reshape((4800, (11-input_shape), 1))
        print(x_vals.shape + y_vals.shape)

        n = 4800
        x_train = x_vals[0:int(0.7*n), :, :]
        y_train = y_vals[0:int(0.7*n), :, :]
        x_test = x_vals[int(0.7*n):, :, :]
        y_test = y_vals[int(0.7*n):, :, :]
        return x_train, y_train, x_test, y_test

    sweep_config = {
    'method': 'bayes', 
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'early_terminate':{
        'type': 'hyperband',
        'min_iter': 1
    },
    'parameters': {
        'batch_size': {
            'values': [4]# [32, 64]
        },
        'learning_rate':{
            'values': [0.0001]#[0.01, 0.001, 0.0001]
        },
        'neurons':{
            'values': [64]#[32, 64, 96, 128]
        },
        'activation':{
            'values': ['tanh']#['sigmoid', 'tanh', 'relu']
        }
    }
    }

    def get_compiled_model(shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(wandb.config.neurons, activation=wandb.config.activation, input_shape=(shape, 1)))
        model.add(tf.keras.layers.RepeatVector(11-shape))
        model.add(tf.keras.layers.LSTM(wandb.config.neurons, activation=wandb.config.activation, return_sequences=True))
        model.add(tf.keras.layers.LSTM(wandb.config.neurons, activation=wandb.config.activation, return_sequences=True))
        model.add(tf.keras.layers.LSTM(wandb.config.neurons, activation=wandb.config.activation, return_sequences=True))
        model.add(tf.keras.layers.LSTM(wandb.config.neurons, activation=wandb.config.activation, return_sequences=True))
        model.add(tf.keras.layers.LSTM(wandb.config.neurons, activation=wandb.config.activation, return_sequences=True))
        model.add(tf.keras.layers.LSTM(wandb.config.neurons, activation=wandb.config.activation, return_sequences=True))
        model.add(tf.keras.layers.LSTM(wandb.config.neurons, activation=wandb.config.activation, return_sequences=True))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
        # model.compile(optimizer='adam', loss='mse')
        return model

    def train():
        # Specify the hyperparameter to be tuned along with
        # an initial value
        config_defaults = {
            'batch_size': 4,
            'learning_rate': 0.0001,
            'neurons': 64,
            'activation': 'tanh'
        }

        # Initialize wandb with a sample project name
        wandb.init(config=config_defaults)

        # Specify the other hyperparameters to the configuration, if any
        wandb.config.epochs = 1

        x_train, y_train, x_test, y_test = data_load(int(args.shape))

        # Prepare trainloader
        trainloader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        trainloader = trainloader.shuffle(1024).batch(wandb.config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        # prepare testloader 
        testloader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        testloader = testloader.batch(wandb.config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # Iniialize model with hyperparameters
        keras.backend.clear_session()
        model = get_compiled_model(int(args.shape))
        
        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate) # optimizer with different learning rate specified by config
        model.compile(opt, metrics=['acc'], loss='mse')
        
        # Train the model
        _ = model.fit(trainloader,
                    epochs=wandb.config.epochs, 
                    validation_data=testloader,
                    callbacks=[WandbCallback()]) # WandbCallback to automatically track metrics
                                
        # Evaluate    
        loss, accuracy = model.evaluate(testloader, callbacks=[WandbCallback()])
        print('Test Error Rate: ', round((1-accuracy)*100, 2))
        wandb.log({'Test Error Rate': round((1-accuracy)*100, 2)}) # wandb.log to track custom metrics


    sweep_id = wandb.sweep(sweep_config, project="ddp-fourth_run", entity="ddp_profpatra")
    wandb.agent(sweep_id, function=train)

if __name__ == '__main__':
    # print("hello")
    main()