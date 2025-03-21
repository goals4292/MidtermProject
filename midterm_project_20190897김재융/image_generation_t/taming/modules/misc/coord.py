import torch

class CoordStage(object):
    def __init__(self, n_embed, down_factor):
        self.n_embed = n_embed
        self.down_factor = down_factor

    def eval(self):
        return self

    def encode(self, c):
        """fake vqmodel interface"""
        assert 0.0 <= c.min() and c.max() <= 1.0
        b,ch,h,w = c.shape
        assert ch == 1

        c = torch.nn.functional.interpolate(c, scale_factor=1/self.down_factor,
                                            mode="area")
        c = c.clamp(0.0, 1.0)
        c = self.n_embed*c
        c_quant = c.round()
        c_ind = c_quant.to(dtype=torch.long)

        info = None, None, c_ind
        return c_quant, None, info

    def decode(self, c):
        c = c/self.n_embed
        c = torch.nn.functional.interpolate(c, scale_factor=self.down_factor,
                                            mode="nearest")
        return c
    
class TopicStage(object):
    def __init__(self, topk, token, embed_dim):
        self.topk = topk
        self.token = token
        self.mult = token//topk
        self.embed_dim = embed_dim

    def eval(self):
        return self

    def encode(self, x):
        """fake vqmodel interface"""
        _, t_ind = torch.topk(x, self.topk, dim=1)
        t_ind = t_ind.repeat(1, self.mult)
        t_ind = t_ind.to(dtype=torch.long)
        return None, t_ind
    
    def decode(self, c):
        c = c/self.n_embed
        c = torch.nn.functional.interpolate(c, scale_factor=self.down_factor,
                                            mode="nearest")
        return c
