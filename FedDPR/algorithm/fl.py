from .base import Algorithm
import numpy as np


class NaiveFl(Algorithm):
    """
    A naive federated learning algorithm without any scoring mechanism.
    """

    def run_a_round(self, t, r):
        """
        Execute a round of federated learning training process.

        :param r: Current training round
        """
        # Print the start of training information
        print("Learner ", end="")
        # Traverse all local learners and perform local training
        for i, learner in enumerate(self.learners):
            # Call the local training method of the local learner
            learner.local_train()
            # Print the index of the current learner
            print(f"{i}..", end="")
        # Print the completion of training information
        print("Finish")

        # Collect the gradients of all local learners
        grads = np.vstack([learner.get_grad() for learner in self.learners])
        # Aggregate the gradients of all local learners
        grads_g = np.vstack(
            [aggregator.aggregate(grads) for aggregator in self.aggregators]
        )

        losses, accs = [], []
        # Update the gradients of all local learners
        for i, learner in enumerate(self.learners):
            learner.set_grads(grads_g)
            loss, acc = learner.test()
            if i >= self.cfg.learner.m:
                # only test benign learners
                loss, acc = learner.test()
                losses.append(loss), accs.append(acc)

        # Calculate the average loss and accuracy of all local learners
        loss, acc = np.mean(losses), np.mean(accs)
        # Print the test results
        print(f"Avg Loss: {loss}, Acc: {acc}")
        # write to db
        self.exec_sql(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
            [self.id, t, r, loss, acc],
        )


class ScoreFl(Algorithm):
    def run_a_round(self, t, r):
        # Print the start of training information
        print("Learner ", end="")
        # Traverse all local learners and perform local training
        for i, learner in enumerate(self.learners):
            # Call the local training method of the local learner
            learner.local_train()
            # Print the index of the current learner
            print(f"{i}..", end="")
        # Print the completion of training information
        print("Finish")

        # Collect the gradients of all local learners
        grads = np.vstack([learner.get_grad() for learner in self.learners])
        # Aggregate the gradients of all local learners
        grads_g = np.vstack(
            [aggregator.aggregate(grads) for aggregator in self.aggregators]
        )

        losses, accs = [], []
        # Update the gradients of all local learners
        for i, learner in enumerate(self.learners):
            learner.update_scores(grads_g)
            learner.set_grads(grads_g)
            if i >= self.cfg.learner.m:
                # only test benign learners
                loss, acc = learner.test()
                losses.append(loss), accs.append(acc)
        # Calculate the average loss and accuracy of all benign learners
        loss, acc = np.mean(losses), np.mean(accs)

        rev_scores = np.vstack(
            [learner.get_rev_scores() for learner in self.learners]
        ).T
        for j, aggregator in enumerate(self.aggregators):
            aggregator.update_scores(rev_scores[j])
        # Print the test results
        print(f"Avg Loss: {loss}, Acc: {acc}")
        # write to db
        self.exec_sql(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
            [self.id, t, r, loss, acc],
        )
        for i, learner in enumerate(self.learners):
            rev_scores = learner.get_rev_scores()
            for j, score in enumerate(rev_scores):
                self.exec_sql(
                    "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?)",
                    [self.id, t, r, "L" + str(i), "A" + str(j), score],
                )

        for j, aggregator in enumerate(self.aggregators):
            scores = aggregator.get_scores()
            for i, score in enumerate(scores):
                self.exec_sql(
                    "INSERT INTO scores VALUES (?,?,?,?,?,?)",
                    [self.id, t, r, "A" + str(j), "L" + str(i), score],
                )
