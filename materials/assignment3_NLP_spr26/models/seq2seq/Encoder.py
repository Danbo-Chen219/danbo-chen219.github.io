"""
S2S Encoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden states of the Encoder(namely, Linear - ReLU - Linear).    #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
            # 1) embedding layer
        self.embedding = nn.Embedding(input_size, emb_size)

        # 2) recurrent layer
        if model_type == "RNN":
            self.rnn = nn.RNN(
                input_size=emb_size,
                hidden_size=encoder_hidden_size, # only the size of the hidden state h_t
                batch_first=True #(batch_size, seq_len, feature_dim)
            )
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=emb_size,
                hidden_size=encoder_hidden_size, # both hidden state h_t and cell state c_t have the same size
                batch_first=True
            )
        else:
            raise ValueError("model_type must be 'RNN' or 'LSTM'")

        # 3) Linear -> ReLU -> Linear
        self.fc1 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(encoder_hidden_size, decoder_hidden_size)

        # 4) dropout
        self.dropout = nn.Dropout(dropout)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the state coming out of the last hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #                                                                           #
        #       Do not apply any tanh (linear layers/Relu) for the cell state when  #
        #       model_type is LSTM before returning it.                             #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################
        embedded = self.embedding(input)       # (N, T, emb_size)
        embedded = self.dropout(embedded)

        if self.model_type == "LSTM":
            output, (hidden, cell) = self.rnn(embedded)

            hidden = self.fc1(hidden)
            hidden = self.relu(hidden)
            hidden = self.fc2(hidden)
            hidden = torch.tanh(hidden)

            # Do NOT modify cell with linear/relu/tanh
            return output, (hidden, cell)

        else:
            output, hidden = self.rnn(embedded)

            hidden = self.fc1(hidden)
            hidden = self.relu(hidden)
            hidden = self.fc2(hidden)
            hidden = torch.tanh(hidden)
        # output, hidden = None, None     #remove this line when you start implementing your code

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

            return output, hidden
