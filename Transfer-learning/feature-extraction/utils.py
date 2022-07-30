#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 12:27:16 2022

@author: markins
"""

import matplotlib.pyplot as plt

def loss_plotter(history):
    loss = history.history["loss"]
    val_loss =  history.history["val_loss"]
    
    accuracy = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    
    epochs = range(len(history
                       .history["loss"]))
    
    plt.plot(epochs, loss, label="Training data")
    plt.plot(epochs, val_loss, label="Testing data")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, accuracy, label="Training data")
    plt.plot(epochs, val_acc, label="Testing data")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    