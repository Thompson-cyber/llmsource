import tensorflow as tf
import keras
from tfdiffeq import odeint
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from time import perf_counter as time
from keras.layers import Dense
from keras.activations import tanh
from keras.initializers import RandomNormal, Zeros

from tools.toydata_processing import val_shift_split
from tools.plots import *
from tools.toydata_processing import get_batch, get_batch_tensorflow
from tools.misc import check_cuda, tictoc, check_cuda_tensorflow

class ODEfunctens(keras.Model):
    
    def __init__(self, N_neurons, N_feat, device):
        super(ODEfunctens, self).__init__()
        self.net = keras.models.Sequential()
        self.net.add(keras.Input(shape=(N_feat,)))
        self.net.add(Dense(N_neurons, activation=tanh, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), bias_initializer=Zeros())) # glorot_normal = initializes weight to normal distribution.
        self.net.add(Dense(N_feat, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), bias_initializer=Zeros()))
        self.net.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                         loss=keras.losses.MeanSquaredError())

        self.device = device
    
    # @tf.function
    def call(self, t, y):
        with tf.device(self.device):
            y = self.net(y)
        return y



def main(num_neurons=50, num_epochs=300, learning_rate=0.01, batch_size=50, batch_dur_idx=20, batch_range_idx=500, rel_tol=1e-7, abs_tol=1e-9, val_freq=5, intermediate_pred_freq=0, live_plot=False):
    """
    Main function for training and evaluating a TensorFlow model using ODE integration.

    Args:
        num_neurons (int): Number of neurons in the ODE function.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        rel_tol (float): Relative tolerance for the ODE solver.
        abs_tol (float): Absolute tolerance for the ODE solver.
        val_freq (int): Frequency of validation evaluation (in terms of epochs).
        intermediate_pred_freq (int): Frequency of intermediate predictions (in terms of epochs). 
            Note: for no intermediate predictions, set this to 0.
        live_plot (bool): Whether to enable live plotting of training and validation losses.
    
    Returns:
        None
    """
    # defining train and validation lists
    train_losses = []
    val_losses = []
    train_loss_cache = []

    # checking if a GPU is available and needed
    available_gpu = tf.config.list_physical_devices('GPU')
    available_cpu = tf.config.list_physical_devices('CPU')
    # print(available_cpu)
    device = 'CPU'

    # data loading
    loaded_data = np.load("NODE/Input_Data/tensors.npz") # needs to become the data for the real data
    data = (tf.convert_to_tensor(loaded_data['t']), tf.convert_to_tensor(loaded_data['features']))
    num_feat = data[1].shape[1]
    with tf.device(device):
        # initialising model, optimizer and loss function
        model = ODEfunctens(num_neurons, num_feat, device=device)
        MSEloss = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        if live_plot:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots(figsize=(10, 6))
            line1, = ax.plot([], [], label='Training Loss')  # Line for training loss
            line2, = ax.plot([], [], label='Validation Loss')  # Line for validation loss
            ax.set_title('Training vs Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            plt.show()
            plt.pause(0.1)

        for epoch in range(num_epochs):

            # get features and timestamps for batches
            t, features = get_batch_tensorflow(data, batch_size=batch_size, batch_dur_idx = batch_dur_idx, batch_range_idx=batch_range_idx, device=device)
            # print("feature shape", features.shape)

            with tf.GradientTape() as tape:
                # make predictions
                y_pred = odeint(model, tf.reshape(features[0],[1,2]), t, rtol=rel_tol, atol=abs_tol) 

                # get the loss value
                loss = MSEloss(y_pred, features)

            # gradient of trainable weights with resprect to loss
            # done using gradient tape.
            grads = tape.gradient(loss, model.trainable_weights)            
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # saving losses
            train_loss_cache.append(loss)

            # validation
            if epoch % val_freq == val_freq-1:
                print("weights are: ", model.trainable_weights)
                print("vars are: ", model.trainable_variables)
                y_pred_val = odeint(model, tf.reshape(data[1][0], [1, num_feat]), data[0])
                val_loss  = MSEloss(y_pred_val, data[1])

                train_losses.append(np.mean(train_loss_cache))
                val_losses.append(val_loss)
                # print(val_loss)
                # print(val_loss.shape)
                # print(type(val_loss))
                print(f"Epoch {epoch+1}: loss = {loss}, val_loss = {val_loss}")


                #live training vs validation plot
                if live_plot:
                    line1.set_data(range(0, epoch + 1, 5), train_losses)
                    line2.set_data(range(0, epoch + 1, 5), val_losses)
                    ax.relim()  # Recalculate limits
                    ax.autoscale_view(True,True,True)  # Autoscale
                    plt.draw()
                    plt.pause(0.3)  # Pause to update the plot
            else:
                print(f"Epoch {epoch+1}: loss = {loss}")
            
            # if intermediate_pred_freq and epoch % intermediate_pred_freq == intermediate_pred_freq-1:
            #         predicted_intermidiate = odeint(model, tf.reshape(data[1][0], [1,2]), data[0])
            #         eval_loss_intermidiate = MSEloss(predicted_intermidiate, data[1])
            #         print(f"Mean Squared Error Loss intermidiate: {eval_loss_intermidiate}")
            #         intermediate_prediction(data, predicted_intermidiate, eval_loss_intermidiate, num_feat, epoch)

        #prediction after training is complete
        # with tf.stop_gradient(model.trainable_weights):
        predicted = odeint(model, tf.reshape(data[1][0], [1,2]), data[0])
        eval_loss = MSEloss(predicted, data[1])
        print(f'"Mean Squared Error Loss: {eval_loss}')
        print(predicted.shape)
    # print(data,shape)
    # Plotting the losses
    toy = True
    for_torch = False
    plot_data(data, toy=toy)
    plot_actual_vs_predicted_full(data, tf.reshape(predicted, [1200,2]), toy=toy, for_torch=for_torch)
    # plot_phase_space(data, tf.reshape(predicted, [1200,2]))
    plot_training_vs_validation([train_losses, val_losses], share_axis=True)
    plt.show()


if __name__ == "__main__":
    # main() # this doesnt work, run from main.py
    pass

# data = torch.load("Input_Data/real_data_scuffed1.pt")
# num_feat = data[1].shape[1]
# print(data[0].shape)
# print(data[1].shape)
# print(num_feat)
# loaded_data = np.load("Input_Data/tensors.npz")
# data = (tf.convert_to_tensor(loaded_data['t']), tf.convert_to_tensor(loaded_data['features']))
# num_feat = data[1].shape[1]
# print(data[0].shape)
# print(data[1].shape)
# print(num_feat)
























# import tensorflow as tf
# from tfdiffeq import odeint
# import pandas as pd
# import numpy as np

# """
# a lot of this garbage is outdated, go look at pytorch versions for more up to date code
# some of the code in this file is replaced by functions that should work in tensorflow aswell i think
# """

# print(f"using TensorFlow version {tf.__version__}")

# data = pd.read_csv("toydatafixed.csv", delimiter=';')
# t_tensor = tf.convert_to_tensor(data['t'].values, dtype=tf.float32)
# features_tensor = tf.convert_to_tensor(data[['speed', 'angle']].values, dtype=tf.float32)
# min_vals = tf.reduce_min(features_tensor, axis=0)
# max_vals = tf.reduce_max(features_tensor, axis=0)
# features_tensor = (features_tensor - min_vals) / (max_vals - min_vals)
# num_feat = features_tensor.shape[1]

# def simple_split(train_dur, val_dur):
#     split_train_dur = train_dur
#     split_val_dur = val_dur

#     split_train = int(split_train_dur / 0.005)  
#     split_val = int((split_val_dur + split_train_dur) / 0.005)
#     train_data = (t_tensor[:split_train], features_tensor[:split_train])
#     val_data = (t_tensor[split_train:split_val], features_tensor[split_train:split_val])
#     test_data = (t_tensor[split_val:], features_tensor[split_val:])

# def val_shift_split(train_dur, val_shift):
#     split_train_dur = train_dur
#     shift_val_dur = val_shift

#     split_train = int(split_train_dur / 0.005)  
#     shift_val = int(shift_val_dur  / 0.005)

#     train_data = (t_tensor[:split_train], features_tensor[:split_train])
#     val_data = (t_tensor[shift_val:split_train + shift_val], features_tensor[shift_val:split_train + shift_val])
#     test_data = (t_tensor[split_train:], features_tensor[split_train:])
#     return train_data, val_data, test_data


# class ODEFunc(tf.keras.Model):

#     def __init__(self, **kwargs):
#         super(ODEFunc, self).__init__(**kwargs)

#         self.x = tf.keras.layers.Dense(50, activation='tanh',
#                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
#         self.y = tf.keras.layers.Dense(2,
#                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

#     def call(self, t, y):
#         y = tf.cast(y, tf.float32)
#         x = self.x(y ** 3)
#         y = self.y(x)
#         return y


# #generated with copilot and not adjust for NODE use
# def train_model(data, data2,lr = 0.01, num_epochs=10):
    
#     t, features = data
#     val_t, val_features = data2


#     loss_fn = tf.keras.losses.MeanSquaredError()
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#     net = ODEFunc()

#     for epoch in range(num_epochs):
#         with tf.GradientTape() as tape:
#             pred_y = odeint(net, features[0], t)
#             loss = loss_fn(pred_y, features)

#         grads = tape.gradient(loss, net.variables)
#         grad_vars = zip(grads, net.variables)
#         optimizer.apply_gradients(grad_vars)


#         print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")



# # train_data, val_data, test_data = simple_split(2, 2)  # Assuming you want to use the simple_split() function
# train_data, val_data, test_data = val_shift_split(3, .3)  # Assuming you want to use the val_shift_split() function

# train_model(train_data, val_data)

