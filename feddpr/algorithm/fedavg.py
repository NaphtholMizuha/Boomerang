from .base import Algorithm, Config


class FedAvg(Algorithm):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        
    def run_inner(self):
        grads = []
        for i, learner in enumerate(self.learners):
            learner.local_train()
            print(f"Learner {i} finished training")
            grads.append(learner.get_weight())

        grad_g = self.aggregator.aggregate(grads)

        for learner in self.learners:
            learner.set_weight(grad_g)

        loss, acc = self.learners[0].test()
        print(f"Loss: {loss}, Acc: {acc}")
        return loss, acc

