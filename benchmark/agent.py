import abc

import jax
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qMaxValueEntropy, qNoisyExpectedImprovement
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.utils import get_optimal_samples
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from jit_env import Action, TimeStep

# import PES
from util import jax_array_to_tensor, tensor_to_jax_array

# use a GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def fit_model(train_x, train_y):
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    # model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


class Agent:
    def __init__(self, agent_name: str, n_dim: int, x_range: [tuple[float, float]], batch_size: int = 1):
        self.agent_name = agent_name
        self.n_dim = n_dim
        # x_min = torch.tensor([x[0] for x in x_range]).double()
        # x_max = torch.tensor([x[1] for x in x_range]).double()
        # self.x_range = torch.stack([x_min, x_max])
        self.x_range = x_range

        self.batch_size = batch_size

        self.actions = None
        self.rewards = None

    def get_name(self):
        return self.agent_name

    @abc.abstractmethod
    def get_action(self,
                   key_policy: jax.random.PRNGKey,
                   observation: tuple[Action, TimeStep],
                   ) -> (Action, ExactGP):
        """
        :param observation:
        :param key_policy:
        :return: Action.
            A Jax-compatible data-structure adhering to self.action_spec().
            Should be the selected point(s), tensor shape (n_dim,).
        """
        raise NotImplementedError

    def reset(self):
        self.actions = None
        self.rewards = None

class BOAgent(Agent):
    def __init__(self, agent_name: str, n_dim: int, x_range: torch.Tensor, batch_size: int = 1):
        super().__init__(agent_name, n_dim, x_range, batch_size)
        self.num_samples = 100
        # self.x_range_normalized = torch.tensor([[0.0 for _ in range(n_dim)], [1.0 for _ in range(n_dim)]])\
        #     .double().to(device)
        self.x_range_normalized = torch.tensor([[0.0 for _ in range(n_dim)], [1.0 for _ in range(n_dim)]]).to(device).double()

    def get_action(self,
                   key_policy: jax.random.PRNGKey,
                   observation: tuple[Action, TimeStep],
                   ) -> (Action, ExactGP):
        actions = jax_array_to_tensor(observation[0])
        rewards = jax_array_to_tensor(observation[1].reward).unsqueeze(-1).to(device)

        normalized_actions = ((actions - self.x_range[0]) / (self.x_range[1] - self.x_range[0])).to(device)

        if self.actions is None:
            self.actions = normalized_actions
            self.rewards = rewards
        else:
            self.actions = torch.cat((self.actions, normalized_actions))
            self.rewards = torch.cat((self.rewards, rewards))

        model = fit_model(self.actions, self.rewards)
        candidate = self.get_candidate(model).to(torch.device('cpu'))

        candidate_unnormalized = candidate * (self.x_range[1] - self.x_range[0]) + self.x_range[0]

        return tensor_to_jax_array(candidate_unnormalized), model

    @abc.abstractmethod
    def get_candidate(self, model):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, agent_name: str, n_dim: int, x_range: [tuple[float, float]], batch_size: int = 1):
        super().__init__(agent_name, n_dim, x_range, batch_size)

    def get_action(self,
                   key_policy: jax.random.PRNGKey,
                   observation: tuple[Action, TimeStep],
                   ) -> (Action, ExactGP):
        return jax.random.uniform(
            key_policy,
            (self.batch_size, self.n_dim),
            minval=tensor_to_jax_array(self.x_range[0]),
            maxval=tensor_to_jax_array(self.x_range[1])
        ), None


# class MyQPES(Agent):
#     def __init__(self, agent_name: str, n_dim: int, x_range: tuple[float, float], batch_size: int = 1):
#         super().__init__(agent_name, n_dim, x_range, batch_size)
#         self.agent = PES.qPredictiveEntropySearch(x_range=self.x_range)
#
#     def get_action(self,
#                    key_policy: jax.random.PRNGKey,
#                    observation: tuple[Action, TimeStep],
#                    ) -> Action:
#         actions = jax_array_to_tensor(observation[0])
#         rewards = jax_array_to_tensor(observation[1].reward)
#         self.agent.update(actions, rewards)
#         return tensor_to_jax_array(self.agent.choose_next())


class qPES(BOAgent):
    def __init__(self, agent_name: str, n_dim: int, x_range: [tuple[float, float]], batch_size: int = 1):
        super().__init__(agent_name, n_dim, x_range, batch_size)

    def get_candidate(self, model):
        optimal_inputs, _ = get_optimal_samples(
            model,
            bounds=self.x_range_normalized,
            num_optima=self.num_samples,
        )

        agent = qPredictiveEntropySearch(model=model, optimal_inputs=optimal_inputs)
        candidate, _ = optimize_acqf(
            acq_function=agent,
            bounds=self.x_range_normalized,
            q=self.batch_size,
            num_restarts=5,
            raw_samples=256,
            options={'with_grad': False},
        )
        return candidate


class qMES(BOAgent):
    def __init__(self, agent_name: str, n_dim: int, x_range: [tuple[float, float]], batch_size: int = 1):
        super().__init__(agent_name, n_dim, x_range, batch_size)

    def get_agent(self, model):
        candidate_set = torch.rand(100*self.n_dim, self.x_range_normalized.size(1), dtype=torch.float64).to(device)
        return qMaxValueEntropy(model=model, candidate_set=candidate_set, )

    def get_candidate(self, model):
        agent = self.get_agent(model)
        candidate, _ = optimize_acqf(
            acq_function=agent,
            bounds=self.x_range_normalized,
            q=self.batch_size,
            num_restarts=5,
            raw_samples=216,
            sequential=True,
        )
        return candidate


class qJES(BOAgent):
    def __init__(self, agent_name: str, n_dim: int, x_range: [tuple[float, float]], batch_size: int = 1):
        super().__init__(agent_name, n_dim, x_range, batch_size)

    def get_agent(self, model, optimal_inputs, optimal_outputs):
        return qJointEntropySearch(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            estimation_type='LB'
        )

    def get_candidate(self, model):
        optimal_inputs, optimal_outputs = get_optimal_samples(
            model,
            bounds=self.x_range_normalized,
            num_optima=self.num_samples,
        )
        agent = self.get_agent(model, optimal_inputs, optimal_outputs)
        candidate, _ = optimize_acqf(
            acq_function=agent,
            bounds=self.x_range_normalized,
            q=self.batch_size,
            num_restarts=5,
            raw_samples=256,
        )
        return candidate

class qEI(BOAgent):
    def __init__(self, agent_name: str, n_dim: int, x_range: [tuple[float, float]], batch_size: int = 1):
        super().__init__(agent_name, n_dim, x_range, batch_size)

    def get_agent(self, model):
        # sampler = SobolQMCNormalSampler(512)
        return qNoisyExpectedImprovement(model, self.actions)

    def get_candidate(self, model):
        agent = self.get_agent(model)
        candidate, _ = optimize_acqf(
            acq_function=agent,
            bounds=self.x_range_normalized,
            q=self.batch_size,
            num_restarts=5,
            raw_samples=216,
        )
        return candidate



