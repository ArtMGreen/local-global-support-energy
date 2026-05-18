import torch
import torch.nn as nn
import torch.nn.functional as F


class T2FNormNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(T2FNormNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.register_buffer('tau', torch.tensor(1.0))

    def set_tau(self, tau):
        self.tau = torch.tensor(tau)

    def forward(self, x, return_with_layers_left=False):
        if return_with_layers_left:
            _, feature_to_return = self.backbone(x, return_with_layers_left=return_with_layers_left)
            _, feature = self.backbone.continue_forward(feature_to_return, layers_left=return_with_layers_left, return_feature=True)
            if self.num_classes != 1000:
                # Imagenet-1k experiment is not trained from scratch.
                # Use temperature scaling only for models trained from scratch.
                feature = feature / self.tau
            output = self.backbone.fc(feature)
            return output, feature_to_return
        else:
            _, feature = self.backbone(x, return_feature=True)
            feature = F.normalize(feature, dim=-1) / self.tau
            output = self.backbone.fc(feature)
            return output

    def continue_forward(self, feature, layers_left, return_feature=False):
        # this line has already happened:
        # _, feature_to_return = self.backbone(x, return_with_layers_left=return_with_layers_left)
        _, feature = self.backbone.continue_forward(feature, layers_left=layers_left, return_feature=True)
        if self.num_classes != 1000:
            # Imagenet-1k experiment is not trained from scratch.
            # Use temperature scaling only for models trained from scratch.
            feature = feature / self.tau
        output = self.backbone.fc(feature)
        if return_feature:
            return output, feature
        return output

    def forward_ood_inference(self, x):
        _, feature = self.backbone(x, return_feature=True)
        if self.num_classes != 1000:
            # Imagenet-1k experiment is not trained from scratch.
            # Use temperature scaling only for models trained from scratch.
            feature = feature / self.tau
        output = self.backbone.fc(feature)
        return output
