import torch.nn as nn

from models.projection_networks import ProjectionLayer
from models.graph_neural_networks import GNNLayer
from models.predict_networks import PredictionLayer


class Model(nn.Module):
    def __init__(self, cfg_model, etypes):
        super(Model, self).__init__()

        self.model_name = cfg_model.name
        self.projection = ProjectionLayer(cfg_model)
        self.gnn = GNNLayer(cfg_model, etypes)
        self.prediction = PredictionLayer(cfg_model)

    def forward(self, pos_graph, neg_graph, blocks):
        blocks = self.projection(blocks)

        x = self.gnn(blocks, blocks[0].srcdata['feature'])

        pos_score = self.prediction(pos_graph, x)
        neg_score = self.prediction(neg_graph, x)

        return pos_score, neg_score

