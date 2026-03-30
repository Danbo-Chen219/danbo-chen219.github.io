"""
LSTM model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # You will need to complete the class init function, and forward function

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # i_t: input gate
        self.W_xi = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        # f_t: the forget gate
        self.W_xf = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        # g_t: the cell gate
        self.W_xg = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b_g = nn.Parameter(torch.zeros(hidden_size))
        # o_t: the output gate
        self.W_xo = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def init_hidden(self, batch_size):
        device = self.W_xi.device
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        self.hidden = (h0, c0)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   h_t and c_t should be initialized to zeros.                                #
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        # h_t, c_t = None, None  #remove this line when you start implementing your code
        batch_size, seq_len, _ = x.shape
        self.init_hidden(batch_size)

        h_prev, c_prev = self.hidden

        for t in range(seq_len):
            x_t = x[:, t, :]   # (batch, input_size)

            i_t = self.sigmoid(x_t @ self.W_xi + h_prev @ self.W_hi + self.b_i)
            f_t = self.sigmoid(x_t @ self.W_xf + h_prev @ self.W_hf + self.b_f)
            g_t = self.tanh(x_t @ self.W_xg + h_prev @ self.W_hg + self.b_g)
            o_t = self.sigmoid(x_t @ self.W_xo + h_prev @ self.W_ho + self.b_o)

            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * self.tanh(c_t)

            h_prev, c_prev = h_t, c_t # previous hidden state and cell state for the next time step

        self.hidden = (h_t, c_t) #So the next time step can access them 
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
