from .dreamer_algo import Dreamer, OptInfo, LossInfo
import torch
from dreamer.utils.module import get_parameters, FreezeParameters
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.tensor import infer_leading_dims
from rlpyt.utils.logging import logger
from torch.utils.tensorboard.writer import SummaryWriter
from rlpyt.replays.sequence.n_step import SamplesFromReplay
from dreamer.algos.replay import samples_to_buffer
from dreamer.models.rnns import get_feat, get_dist
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


class DreamerDBC(Dreamer):
    def __init__(self, bisim_coef: float = 50.0, **kwargs):
        super().__init__(**kwargs)
        self.bisim_coef = bisim_coef

    def optim_initialize(self, rank=0):
        # TODO might still want to split encoder optimizer and models optimizer
        self.rank = rank
        model = self.agent.model
        self.model_modules = [
            model.observation_encoder,
            model.reward_model,
            model.representation,  # TODO what does this do??
            model.transition]
        if self.use_pcont:
            self.model_modules += [model.pcont]
        self.actor_modules = [model.action_decoder]
        self.value_modules = [model.value_model]
        self.model_optimizer = torch.optim.Adam(
            get_parameters(self.model_modules), lr=self.model_lr,
            **self.optim_kwargs)
        self.actor_optimizer = torch.optim.Adam(
            get_parameters(self.actor_modules), lr=self.actor_lr,
            **self.optim_kwargs)
        self.value_optimizer = torch.optim.Adam(
            get_parameters(self.value_modules), lr=self.value_lr,
            **self.optim_kwargs)

        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)
        # must define these fields to for logging purposes. Used by runner.
        self.opt_info_fields = OptInfo._fields

    def loss(self, samples: SamplesFromReplay, sample_itr: int, opt_itr: int):
        """
        Compute the loss for a batch of data.  This includes computing the model and reward losses on the given data,
        as well as using the dynamics model to generate additional rollouts, which are used for the actor and value
        components of the loss.
        :param samples: samples from replay
        :param sample_itr: sample iteration
        :param opt_itr: optimization iteration
        :return: FloatTensor containing the loss
        """
        model = self.agent.model

        # [t, t+batch_length+1] -> [t, t+batch_length]
        observation = samples.all_observation[:-1]
        # [t-1, t+batch_length] -> [t, t+batch_length]
        action = samples.all_action[1:]
        # [t-1, t+batch_length] -> [t, t+batch_length]
        reward = samples.all_reward[1:]
        reward = reward.unsqueeze(2)
        done = samples.done
        done = done.unsqueeze(2)

        # Extract tensors from the Samples object
        # They all have the batch_t dimension first, but we'll put the batch_b dimension first.
        # Also, we convert all tensors to floats so they can be fed into our models.

        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(
            observation, 3)
        # squeeze batch sizes to single batch dimension for imagination roll-out
        batch_size = batch_t * batch_b

        # normalize image
        observation = observation.type(self.type) / 255.0 - 0.5
        # embed the image
        embed = model.observation_encoder(
            observation)  # TODO make this probabilistic

        prev_state = model.representation.initial_state(
            batch_b, device=action.device, dtype=action.dtype)
        # Rollout model by taking the same series of actions as the real model
        prior, post = model.rollout.rollout_representation(
            batch_t, embed, action, prev_state)
        # Flatten our data (so first dimension is batch_t * batch_b = batch_size)
        # since we're going to do a new rollout starting from each state visited in each batch.

        # Compute losses for each component of the model

        # Model Loss
        feat = get_feat(post)
        reward_pred = model.reward_model(feat)
        reward_loss = -torch.mean(reward_pred.log_prob(reward))
        pcont_loss = torch.tensor(0.)  # placeholder if use_pcont = False
        if self.use_pcont:
            pcont_pred = model.pcont(feat)
            pcont_target = self.discount * (1 - done.float())
            pcont_loss = -torch.mean(pcont_pred.log_prob(pcont_target))
        prior_dist = get_dist(prior)
        post_dist = get_dist(post)
        div = torch.mean(
            torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        div = torch.max(div, div.new_full(div.size(), self.free_nats))
        model_loss = self.kl_scale * div + reward_loss
        if self.use_pcont:
            model_loss += self.pcont_scale * pcont_loss
        # Bisim Loss

        bisim_loss = self.compute_bisim_loss(
            observation=observation, action=action)

        # Bisimulation Loss

        # ------------------------------------------  Gradient Barrier  ------------------------------------------------
        # Don't let gradients pass through to prevent overwriting gradients.
        # Actor Loss

        # remove gradients from previously calculated tensors
        with torch.no_grad():
            if self.use_pcont:
                # "Last step could be terminal." Done in TF2 code, but unclear why
                flat_post = buffer_method(
                    post[:-1, :], 'reshape', (batch_t - 1) * (batch_b), -1)
            else:
                flat_post = buffer_method(post, 'reshape', batch_size, -1)
        # Rollout the policy for self.horizon steps. Variable names with imag_ indicate this data is imagined not real.
        # imag_feat shape is [horizon, batch_t * batch_b, feature_size]
        with FreezeParameters(self.model_modules):
            imag_dist, _ = model.rollout.rollout_policy(
                self.horizon, model.policy, flat_post)

        # Use state features (deterministic and stochastic) to predict the image and reward
        # [horizon, batch_t * batch_b, feature_size]
        imag_feat = get_feat(imag_dist)
        # Assumes these are normal distributions. In the TF code it's be mode, but for a normal distribution mean = mode
        # If we want to use other distributions we'll have to fix this.
        # We calculate the target here so no grad necessary

        # freeze model parameters as only action model gradients needed
        with FreezeParameters(self.model_modules + self.value_modules):
            imag_reward = model.reward_model(imag_feat).mean
            value = model.value_model(imag_feat).mean
        # Compute the exponential discounted sum of rewards
        if self.use_pcont:
            with FreezeParameters([model.pcont]):
                discount_arr = model.pcont(imag_feat).mean
        else:
            discount_arr = self.discount * torch.ones_like(imag_reward)
        returns = self.compute_return(imag_reward[:-1], value[:-1], discount_arr[:-1],
                                      bootstrap=value[-1], lambda_=self.discount_lambda)
        # Make the top row 1 so the cumulative product starts with discount^0
        discount_arr = torch.cat(
            [torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        actor_loss = -torch.mean(discount * returns)

        # ------------------------------------------  Gradient Barrier  ------------------------------------------------
        # Don't let gradients pass through to prevent overwriting gradients.
        # Value Loss

        # remove gradients from previously calculated tensors
        with torch.no_grad():
            value_feat = imag_feat[:-1].detach()
            value_discount = discount.detach()
            value_target = returns.detach()
        value_pred = model.value_model(value_feat)
        log_prob = value_pred.log_prob(value_target)
        value_loss = -torch.mean(value_discount * log_prob.unsqueeze(2))

        # ------------------------------------------  Gradient Barrier  ------------------------------------------------
        # loss info
        with torch.no_grad():
            prior_ent = torch.mean(prior_dist.entropy())
            post_ent = torch.mean(post_dist.entropy())
            loss_info = LossInfo(model_loss, actor_loss, value_loss, prior_ent, post_ent, div, reward_loss, bisim_loss,
                                 pcont_loss)

            if self.log_video:
                if opt_itr == self.train_steps - 1 and sample_itr % self.video_every == 0:
                    self.write_videos(
                        observation,
                        step=sample_itr,
                        n=self.video_summary_b)

        return (model_loss + bisim_loss), actor_loss, value_loss, loss_info

    def compute_bisim_loss(self, observation, action):
        """
        observation: torch.Tensor B x T x C x H x W
        action: torch.Tensor B x A
        """
        model = self.agent.model
        observation = torch.cat(tuple(observation), dim=0)
        action = torch.cat(tuple(action), dim=0)

        # Sample random states across episodes at random
        batch_size = observation.size(0)
        perm = np.random.permutation(batch_size)

        with torch.no_grad():
            # Turn into RSSMState for transition model
            state1 = model.get_state_representation(observation)
            # TODO maybe uncomment out next line?
            # action, _ = model.policy(state)
            reward1 = model.reward_model(get_feat(state1))
            next_state1 = model.transition(action, state1)
        reward2_mean = reward1.mean[perm]
        # reward2_variance = reward1.variance[perm]  # TODO probabilistic rewards model difference?
        state2 = state1[perm]
        next_state2 = next_state1[perm]

        # z_dist = F.smooth_l1_loss(h, h2, reduction='none')
        z_dist = torch.sqrt(
            (state1.mean - state2.mean).pow(2) +
            (state1.std - state2.std).pow(2))
        r_dist = F.smooth_l1_loss(
            reward1.mean, reward2_mean, reduction='none')
        transition_dist = torch.sqrt(
            (next_state1.mean - next_state2.mean).pow(2) +
            (next_state1.std - next_state2.std).pow(2))

        bisimilarity = r_dist + self.discount * transition_dist
        loss = (z_dist - bisimilarity).pow(2).mean()
        return loss

    def write_videos(self, observation, step=None, n=4):
        video = torch.clamp(observation[:, :n] + 0.5, 0., 1.).transpose(1, 0)
        writer: SummaryWriter = logger.get_tf_summary_writer()
        writer.add_video(tag='videos/ground_truth',
                         vid_tensor=video,
                         global_step=step,
                         fps=20)
