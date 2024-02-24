import copy


class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model, model2=None, beta=None):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        for i, name in enumerate(self.param_keys):
            if model2 and beta:
                if i < len(self.param_keys):
                    state2 = model2.state_dict()
                    self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * (state[name] * beta + state2[name] * (1-beta)))
            else:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for i, name in enumerate(self.buffer_keys):
            if self.buffer_ema:
                if model2 and beta:
                    if i < len(self.buffer_keys):
                        state2 = model2.state_dict()
                        self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * (state[name] * beta + state2[name] * (1-beta)))
                else:
                    self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }
