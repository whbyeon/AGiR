"""Based on PureJaxRL and JaxMARL. LASE.
Uses other agents' actual observations instead of an imagination module."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import pdb


class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Imagine(nn.Module):

    @nn.compact
    def __call__(self, obs, other_id_onehot):  # obs: (B, C, H, W)
        B, C, H, W = obs.shape
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(obs)  # (B, 16, H, W)
        x = x.reshape((x.shape[0], -1))                                    # (B, 16*H*W)
        x = jnp.concatenate([x, other_id_onehot], axis=-1)                 # (B, 16*H*W + A)
        x = nn.relu(nn.Dense(features=C * H * W)(x))  # (B, C*H*W)
        x = nn.sigmoid(nn.Dense(features=C * H * W)(x))  # (B, C*H*W)
        x = x.reshape((x.shape[0], C, H, W))  # (B, C, H, W)
        return x


class SRPolicy(nn.Module):      # Estimate the best-response policy against the current opponents.
    action_dim: int

    @nn.compact
    def __call__(self, obs):  # obs: (B, C, H, W)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(obs)  # (B, 16, H, W)
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(features=64)(x))
        logits = nn.Dense(features=self.action_dim)(x)
        return distrax.Categorical(logits=logits)


class SRQNet(nn.Module):        # Estimate the best-response Q-function against the current opponents.

    @nn.compact
    def __call__(self, obs, joint_action):  # obs: (B, C, H, W)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(obs)
        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, joint_action], axis=-1)
        x = nn.relu(nn.Dense(features=64)(x))
        q = nn.Dense(features=1)(x)
        return jnp.squeeze(q, axis=-1)  # (B,)


def compute_gifting_weights_single(i, obs, joint_action,
                                   sr_q_apply, sr_q_params_i,
                                   sr_policy_apply, sr_policy_params_i,
                                   imagine_apply, imagine_params_i,
                                   agent_ids_onehot, config):
    """
    Compute gifting weights from agent i to all other agents j ≠ i.

    Args:
        i: int. index of focal agent (giver)
        obs: (B, A, C, H, W)  # agent-wise observations
        joint_action: (B, A * action_dim)  # joint one-hot action vector
        sr_q_apply: function for sr_q network
        sr_q_params_i: params for agent i's sr_q
        sr_policy_apply: function for sr_policy network
        sr_policy_params_i: params for agent i's sr_policy
        imagine_apply: function for imagine model
        imagine_params_i: params for agent i's imagine model
        agent_ids_onehot: (A, A)  # one-hot encoding of each agent ID
        config: dict with "num_agents", "action_dim"

    Returns:
        weights: (B, A), where weights[:, j] is the weight agent i assigns to agent j
    """

    B, A = obs.shape[0], config["num_agents"]
    action_dim = config["action_dim"]

    obs_i = obs[:, i, ...]  # (B, C, H, W)

    def _per_j(j):
        def compute_for_other():
            # (B, A, action_dim) → (B, A, action_dim)
            a_joint = joint_action.reshape(B, A, action_dim)  # (B, A, action_dim)

            o_hat_j = obs[:, j, ...]    # Use other agents' actual observations instead of an imagination model.

            pi_sr = sr_policy_apply(sr_policy_params_i, o_hat_j)
            a_j_probs = pi_sr.probs  # (B, action_dim)

            def integrate_counterfactual(a_j_prime):
                a_new = a_joint.at[:, j].set(nn.one_hot(a_j_prime, action_dim))  # (B, A, action_dim)
                return sr_q_apply(sr_q_params_i, obs_i, a_new.reshape(B, -1))  # (B,)

            a_j_vals = jnp.arange(action_dim)
            counterfactual_qs = jax.vmap(integrate_counterfactual)(a_j_vals)  # (action_dim, B)
            expected_q = jnp.sum(a_j_probs.T * counterfactual_qs, axis=0)     # (B,)
            q_max = jnp.max(counterfactual_qs, axis=0)
            q_min = jnp.min(counterfactual_qs, axis=0)
            denom = (A - 1) * (q_max - q_min + 1e-8)

            true_q = sr_q_apply(sr_q_params_i, obs_i, joint_action)  # (B,)
            raw = (true_q - expected_q) / denom
            weight = jnp.maximum(raw, 0.0)
            return weight  # (B,)

        return jax.lax.cond(i == j, lambda _: jnp.zeros(B), lambda _: compute_for_other(), operand=None)

    weights = jnp.stack([_per_j(j) for j in range(A)], axis=1)  # (B, A)

    return weights  # weights[:, j] transfers reward from giver i to receiver j.


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    gift_reward: jnp.ndarray
    gift_given: jnp.ndarray


def get_rollout(params, config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *(env.observation_space()[0]).shape)

        env_act = {}
        for i in range(env.num_agents):
            pi, value = network[i].apply(params[i], obs_batch)
            action = pi.sample(seed=key_a0)
            env_act[env.agents[i]] = action

        env_act = {k: v.squeeze() for k, v in env_act.items()}


        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    config["action_dim"] = env.action_space().n

    env = LogWrapper(env, replace_info=False)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=0.,
        end_value=1.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    rew_shaping_anneal_org = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # Initialize policy and value networks.
        network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        network_params = [network[i].init(_rng, init_x) for i in range(env.num_agents)]

        # Initialize LASE social-relationship modules.
        dummy_id = jnp.zeros((1, env.num_agents))
        dummy_action = jnp.zeros((1, env.num_agents * env.action_space().n))

        imagine_model = [Imagine() for _ in range(env.num_agents)]
        sr_policy_model = [SRPolicy(env.action_space().n) for _ in range(env.num_agents)]
        sr_q_model = [SRQNet() for _ in range(env.num_agents)]

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        imagine_params = [imagine_model[i].init(rng1, init_x, dummy_id) for i in range(env.num_agents)]
        sr_policy_params = [sr_policy_model[i].init(rng2, init_x) for i in range(env.num_agents)]
        sr_q_params = [sr_q_model[i].init(rng3, init_x, dummy_action) for i in range(env.num_agents)]


        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            sr_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            im_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            sr_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR_SR"], eps=1e-5),
            )
            im_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR_IMAGINE"], eps=1e-5),
            )
        train_state = [TrainState.create(
            apply_fn=network[i].apply,
            params=network_params[i],
            tx=tx,
        ) for i in range(env.num_agents)]
        sr_q_state = [TrainState.create(
            apply_fn=sr_q_model[i].apply,
            params=sr_q_params[i],
            tx=sr_tx,
        ) for i in range(env.num_agents)]
        sr_policy_state = [TrainState.create(
            apply_fn=sr_policy_model[i].apply,
            params=sr_policy_params[i],
            tx=sr_tx,
        ) for i in range(env.num_agents)]
        im_train_state = [TrainState.create(
            apply_fn=imagine_model[i].apply,
            params=imagine_params[i],
            tx=im_tx,
        ) for i in range(env.num_agents)]

        # Initialize parallel environments.
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # Training loop.
        def _update_step(runner_state, unused):
            # Collect rollout transitions.
            def _env_step(runner_state, unused):
                train_state, sr_q_state, sr_policy_state, im_train_state, env_state, last_obs, update_step, rng = runner_state


                rng, _rng = jax.random.split(rng)


                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))                   # (A, E, obs_dim)
                env_act = {}
                log_prob = []
                value = []
                for i in range(env.num_agents):
                    print(f"input_obs_shape_agent{i}", obs_batch[i].shape)       # (E, obs_dim)
                    pi, value_i = network[i].apply(train_state[i].params, obs_batch[i]) # (E, action_dim)
                    action = pi.sample(seed=_rng)           # (E,)
                    log_prob.append(pi.log_prob(action))
                    env_act[env.agents[i]] = action         # action: (E,)
                    value.append(value_i)                   # List of arrays, each (E,)

                env_act = [v for v in env_act.values()]


                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)


                num_envs = config["NUM_ENVS"]
                num_agents = env.num_agents
                action_dim = env.action_space().n
                num_actors = config["NUM_ACTORS"]

                obs_tensor = obsv  # (E,A,obs_dim)

                action = jnp.stack(env_act).T         # (E,A)
                action_onehot = jax.nn.one_hot(action, action_dim)  # (E,A,action_dim)
                joint_action = action_onehot.reshape(num_envs, num_agents * action_dim)     # (E,A*action_dim)

                agent_ids_onehot = jnp.eye(num_agents)

                # compute gifting weights: (E, A, A)
                weights = []
                for i in range(env.num_agents):
                    weights_i = compute_gifting_weights_single(
                        i=i,
                        obs=obs_tensor,
                        joint_action=joint_action,
                        sr_q_apply=sr_q_model[i].apply,
                        sr_q_params_i=sr_q_params[i],
                        sr_policy_apply=sr_policy_model[i].apply,
                        sr_policy_params_i=sr_policy_params[i],
                        imagine_apply=imagine_model[i].apply,
                        imagine_params_i=imagine_params[i],
                        agent_ids_onehot=agent_ids_onehot,
                        config={"num_agents": num_agents, "action_dim": action_dim},
                    )
                    weights.append(weights_i)
                weights = jnp.stack(weights, axis=1)    # (E, giver, receiver)


                diag = 1.0 - jnp.sum(weights, axis=-1)  # (E, A)
                weights = weights.at[:, jnp.arange(num_agents), jnp.arange(num_agents)].set(diag)

                reward_tensor = reward

                # Received gift: sum of incoming transfers from other agents.
                weights_no_diag = weights.at[:, jnp.arange(num_agents), jnp.arange(num_agents)].set(0.0)
                gift_reward = jnp.einsum("eij,ej->ei", weights_no_diag.transpose((0, 2, 1)), reward_tensor)  # (E,A)

                # Given gift: the outgoing fraction of the agent's own reward.
                w_ii = jnp.einsum("eii->ei", weights)  # (E, A)
                gift_given = (1.0 - w_ii) * reward_tensor    # (E, A)


                transition = []
                done = [v for v in done.values()]
                for i in range(env.num_agents):
                    info_i = {key: jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"]),1), value[:,i]) for key, value in info.items()}
                    # Without parameter sharing, each agent processes E environments.
                    # info_i: (E,A)->(E,)->(E,1)
                    transition.append(Transition(
                        done[i],
                        env_act[i],
                        value[i],
                        reward[:,i],        # (E,A)->(E,)
                        log_prob[i],
                        obs_batch[i],
                        info_i,                 # (E,1)
                        gift_reward[:,i],        # (E,)
                        gift_given[:,i],          # (E,)
                    ))


                runner_state = (train_state, sr_q_state, sr_policy_state, im_train_state, env_state, obsv, update_step, rng)
                return runner_state, transition


            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # Compute GAE.
            train_state, sr_q_state, sr_policy_state, im_train_state, env_state, last_obs, update_step, rng = runner_state

            last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))    # (A,E,obs_dim)
            last_val = []
            for i in range(env.num_agents):
                _, last_val_i = network[i].apply(train_state[i].params, last_obs_batch[i])
                last_val.append(last_val_i)
            last_val = jnp.stack(last_val, axis=0)  # (A,E)

            T = traj_batch[0].obs.shape[0]
            E = traj_batch[0].obs.shape[1]
            action_dim = env.action_space().n

            joint_action_next = []
            for i in range(env.num_agents):
                pi_next, _ = network[i].apply(train_state[i].params, last_obs_batch[i])    # Policy batch: E environments.
                rng, rng_action = jax.random.split(rng)
                action_next = pi_next.sample(seed=rng_action)           # (E,)
                joint_action_next.append(action_next)
            joint_action_next = jnp.stack(joint_action_next, axis=-1)
            action_next_onehot = jax.nn.one_hot(joint_action_next, action_dim).reshape((E, -1))      # (E,A*action_dim)

            td_target_sr = []
            td_error_sr = []
            obs_dim = traj_batch[0].obs.shape[2:]
            for i in range(env.num_agents):

                joint_action = jnp.stack([traj_batch[j].action for j in range(env.num_agents)], axis=-1)   # (T,E,A)
                flat_joint_action = joint_action.reshape((-1, env.num_agents))      # (T*E, A)
                action_onehot = jax.nn.one_hot(flat_joint_action, action_dim).reshape((T*E, -1))  # (T*E,A,action_dim) -> (T*E,A*action_dim)
                flat_obs = traj_batch[i].obs.reshape((T*E, *obs_dim))     # (T*E,obs_dim)
                flat_q_sr_now = sr_q_state[i].apply_fn(sr_q_state[i].params, flat_obs, action_onehot)  # (T*E,)
                q_sr_last = sr_q_state[i].apply_fn(sr_q_state[i].params, last_obs_batch[i], action_next_onehot)  # (E,)
                q_sr_now = flat_q_sr_now.reshape((T,E))


                q_sr_next = jnp.concatenate([
                    q_sr_now[1:, :],         # (T-1,E)
                    q_sr_last[None, ...], # (1,E)
                ], axis=0)  # shape: (T, E)


                td_target_sr_i = traj_batch[i].reward + config["GAMMA"] * q_sr_next * (1 - traj_batch[i].done)    # (T,E)
                td_error_sr_i = td_target_sr_i - q_sr_now                   # (T,E)
                td_target_sr_i = jax.lax.stop_gradient(td_target_sr_i)
                td_error_sr_i = jax.lax.stop_gradient(td_error_sr_i)    # Use the SR advantage as the social-relationship policy-gradient weight.
                td_target_sr.append(td_target_sr_i)
                td_error_sr.append(td_error_sr_i)


            # Compute GAE using the gift-shaped reward.
            def _calculate_gae(traj_batch, last_val, update_step):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                def _use_extrinsic_reward(transition):
                    return transition.reward        # (T,E)

                def _use_shaped_reward(transition):
                    return transition.reward - transition.gift_given + transition.gift_reward       # (T,E)

                shaped_reward = jax.lax.cond(update_step < config["LASE_KWARGS"]["start_learn_gift"], _use_extrinsic_reward, _use_shaped_reward, operand=traj_batch)
                updated_transition = traj_batch._replace(reward=shaped_reward)

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    updated_transition,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages = []
            targets = []
            for i in range(env.num_agents):
                advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val[i], update_step)
                advantages.append(advantages_i)
                targets.append(targets_i)
            advantages = jnp.stack(advantages, axis=0)      # (A, T, E)
            targets = jnp.stack(targets, axis=0)            # (A, T, E)


            # Update policy and value networks.

            def _update_epoch(update_state, unused, i):
                def _update_minbatch(train_state, batch_info, network_used):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, network_used):

                        pi, value = network_used.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)


                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )


                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets, network_used
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state_dict = []
            loss_info_all = []
            for i in range(env.num_agents):
                update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                )
                update_state_dict.append(update_state)
                train_state[i] = update_state[0]
                loss_info_all.append(loss_info[0])

                rng = update_state[-1]
            loss_info = jnp.mean(jnp.stack(loss_info_all))

            # Update LASE social-relationship modules.
            def _train_sr_modules(
                traj_batch,
                td_target_sr,
                td_error_sr,
                sr_q_state,
                sr_policy_state,
                config,
                i,
            ):
                """
                Train SR moduls for agent i.
                Update SR modules with extrinsic rewards: PG loss + TD loss.
                Uses TrainState with tx_sr and repeated gradient updates.
                Assumes traj_batch[j].{obs, action, ...} are shape (T, E, ...) for all j
                """
                action_dim=config["action_dim"]
                num_agents = config["num_agents"]
                num_envs = config["num_envs"]
                grad_steps = config["grad_steps"]

                T = traj_batch[0].obs.shape[0]

                # Train the SR Q-network.
                def _sr_q_loss_fn(params):
                    joint_action = jnp.stack([traj_batch[j].action for j in range(num_agents)], axis=-1).reshape((-1, num_agents))    # (T,E,A)->(T*E,A)
                    joint_action_onehot = jax.nn.one_hot(joint_action, action_dim).reshape((T*num_envs, -1))      # (T*E,A,action_dim)->(T*E,A*action_dim)
                    obs_dim = traj_batch[i].obs.shape[2:]
                    flat_obs = traj_batch[i].obs.reshape((T*num_envs, *obs_dim))          # (T*E, obs_dim)
                    q_pred = sr_q_state.apply_fn(params, flat_obs, joint_action_onehot)        # (T*E,)
                    flat_td_target_sr = td_target_sr.reshape(-1)    # (T,E)->(T*E,)
                    loss = jnp.mean((q_pred - jax.lax.stop_gradient(flat_td_target_sr)) ** 2)
                    return loss

                def _sr_q_update_step(i, state_tuple):
                    params, opt_state = state_tuple
                    loss, grads = jax.value_and_grad(_sr_q_loss_fn)(params)
                    updates, new_opt_state = sr_tx.update(grads, opt_state, params)
                    new_params = optax.apply_updates(params, updates)
                    return (new_params, new_opt_state)

                params, opt_state = sr_q_state.params, sr_q_state.opt_state
                params, opt_state = jax.lax.fori_loop(0, grad_steps, _sr_q_update_step, (params, opt_state))
                sr_q_state = sr_q_state.replace(params=params, opt_state=opt_state)


                # Train the SR policy.
                def _sr_policy_loss_fn(params):
                    obs_dim = traj_batch[i].obs.shape[2:]
                    flat_obs = traj_batch[i].obs.reshape((T*num_envs, *obs_dim))          # (T*E, obs_dim)
                    pi_sr = sr_policy_state.apply_fn(params, flat_obs)          # (T*E, action_dim)
                    flat_action = traj_batch[i].action.reshape(-1)          # (T*E, )
                    log_prob = pi_sr.log_prob(flat_action)     # (T*E,)
                    flat_td_error_sr = td_error_sr.reshape(-1)
                    loss = -jnp.mean(jax.lax.stop_gradient(flat_td_error_sr) * log_prob)
                    return loss

                def _sr_policy_update_step(i, state_tuple):
                    params, opt_state = state_tuple
                    loss, grads = jax.value_and_grad(_sr_policy_loss_fn)(params)
                    updates, new_opt_state = sr_tx.update(grads, opt_state, params)
                    new_params = optax.apply_updates(params, updates)
                    return (new_params, new_opt_state)

                params, opt_state = sr_policy_state.params, sr_policy_state.opt_state
                params, opt_state = jax.lax.fori_loop(0, grad_steps, _sr_policy_update_step, (params, opt_state))
                sr_policy_state = sr_policy_state.replace(params=params, opt_state=opt_state)


                sr_loss = _sr_q_loss_fn(sr_q_state.params) + _sr_policy_loss_fn(sr_policy_state.params)

                return sr_q_state, sr_policy_state, sr_loss


            sr_loss = []

            for i in range(env.num_agents):
                sr_q_state_i, sr_policy_state_i, sr_loss_i = _train_sr_modules(
                    traj_batch=traj_batch,
                    td_target_sr=td_target_sr[i],      # (T,E)
                    td_error_sr=td_error_sr[i],         # (T,E)
                    sr_q_state=sr_q_state[i],
                    sr_policy_state=sr_policy_state[i],
                    config={"action_dim": config["action_dim"], "num_agents": env.num_agents, "num_envs": config["NUM_ENVS"], "grad_steps": config["GRAD_STEPS"]},
                    i=i,
                )
                sr_q_state[i] = sr_q_state_i
                sr_policy_state[i] = sr_policy_state_i
                sr_loss.append(sr_loss_i)


            sr_loss = jnp.mean(jnp.stack(sr_loss))


            metric = {}
            for i in range(env.num_agents):
                metric_i = dict(traj_batch[i].info)

                for k, v in metric_i.items():
                    metric[f"agent{i}_{k}"] = v

            metric["LASE AC loss"] = loss_info
            metric["LASE SR loss"] = sr_loss

            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1

            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)


            num_steps = traj_batch[0].reward.shape[0]
            num_envs = config["NUM_ENVS"]

            returned_episode_returns_average = 0.0
            returns_using_reward_average = 0.0
            for i in range(env.num_agents):
                returned_episode_returns = traj_batch[i].info["returned_episode_returns"].reshape((num_steps, num_envs))    # (T,E,1)->(T,E)
                returned_episode_returns_per_episode = returned_episode_returns[-1]                     # (E,)
                returned_episode_returns_per_agent = jnp.mean(returned_episode_returns_per_episode)
                metric[f"agent{i}_env_return"] = returned_episode_returns_per_agent
                returned_episode_returns_average = returned_episode_returns_average + returned_episode_returns_per_agent

                reward_from_traj_batch = traj_batch[i].reward    # (T,E)
                episode_returns_per_episode_using_reward = jnp.sum(reward_from_traj_batch, axis=0)                    # (E,)
                episode_returns_per_agent_using_reward = jnp.mean(episode_returns_per_episode_using_reward)
                metric[f"agent{i}_received_reward_return"] = episode_returns_per_agent_using_reward      # env is already giving "shaped" reward
                returns_using_reward_average = returns_using_reward_average + episode_returns_per_agent_using_reward

                gift_reward = traj_batch[i].gift_reward  # (T,E)
                gift_return_per_episode = jnp.sum(gift_reward, axis=0)  # shape: (E,)
                gift_return_mean_per_agent = jnp.mean(gift_return_per_episode)
                metric[f"agent{i}_received_gift_return"] = gift_return_mean_per_agent

                gift_given = traj_batch[i].gift_given    # shape (T, E)
                gift_given_per_episode = jnp.sum(gift_given, axis=0)  # shape: (E,)
                gift_given_mean_per_agent = jnp.mean(gift_given_per_episode)
                metric[f"agent{i}_given_gift_return"] = gift_given_mean_per_agent

                metric[f"agent{i}_total_return"] = (
                    returned_episode_returns_per_agent + gift_return_mean_per_agent - gift_given_mean_per_agent
                )
                metric[f"agent{i}_total_return_using_reward"] = (
                    episode_returns_per_agent_using_reward + gift_return_mean_per_agent - gift_given_mean_per_agent
                )
            metric["average_env_return"] = returned_episode_returns_average/env.num_agents
            metric["average_env_return_using_reward"] = returns_using_reward_average/env.num_agents


            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            for i in range(env.num_agents):
                metric[f"agent{i}_AppleCount_info"] = metric[f"agent{i}_AppleCount_info"] * config["ENV_KWARGS"]["num_inner_steps"]
            jax.debug.callback(callback, metric)

            runner_state = (train_state, sr_q_state, sr_policy_state, im_train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, sr_q_state, sr_policy_state, im_train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

def single_run(config):
    config = OmegaConf.to_container(config)


    wandb.login()
    group_name = f'env={str(config["ENV_NAME"])[:9]}_4a_{config["MAP_NAME"]}_algo=LASE-NI_slg={config["LASE_KWARGS"]["start_learn_gift"]}_nb={config["NUM_MINIBATCHES"]}_lr={config["LR"]}_num_envs={config["NUM_ENVS"]}_num_steps={config["NUM_STEPS"]}_envsteps={config["ENV_KWARGS"]["num_inner_steps"]}_gradsteps={config["GRAD_STEPS"]}'
    run_name=f'4a_{config["MAP_NAME"]} seed={config["SEED"]} slg={config["LASE_KWARGS"]["start_learn_gift"]} harvest lase no img,gradsteps={config["GRAD_STEPS"]} {config["NUM_ENVS"]},{config["NUM_STEPS"]},{config["ENV_KWARGS"]["num_inner_steps"]}'
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=run_name,
        group=group_name,
        job_type="harvest",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'lase-{config["MAP_NAME"]}_seed{config["SEED"]}'


    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])


    params = []
    for i in range(config['ENV_KWARGS']['num_agents']):
        save_path = f"./checkpoints/individual/{filename}_{i}.pkl"
        save_params(train_state[i], save_path)
        params.append(load_params(save_path))

    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)


def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)

def evaluate(params, env, save_path, config):
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/{config['ENV_NAME']}_4a_{config['MAP_NAME']}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):

        obs_batch = jnp.stack([obs[a] for a in env.agents])
        env_act = {}
        network = [ActorCritic(action_dim=env.action_space().n, activation="relu") for _ in range(env.num_agents)]
        for i in range(env.num_agents):
            obs = jnp.expand_dims(obs_batch[i],axis=0)
            pi, _ = network[i].apply(params[i], obs)
            rng, _rng = jax.random.split(rng)
            single_action = pi.sample(seed=_rng)
            env_act[env.agents[i]] = single_action


        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]


        img = env.render(state)
        pics.append(img)

        print('###################')
        print(f'Actions: {env_act}')
        print(f'Reward: {reward}')

        print("###################")


    from datetime import datetime
    import pytz


    korea_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(korea_tz)

    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

    gif_path = f'{root_dir}/algo=LASE_{formatted_time}_frame_{o_t+1}_gradsteps={config["GRAD_STEPS"]}_nb={config["NUM_MINIBATCHES"]}_lr={config["LR"]}_num_steps={config["NUM_STEPS"]}_envsteps={config["ENV_KWARGS"]["num_inner_steps"]}_seed={config["SEED"]}.gif'


    print(f"Saving Episode GIF")
    pics = [Image.fromarray(img) for img in pics]
    pics[0].save(

    gif_path,
    format="GIF",
    save_all=True,
    optimize=False,
    append_images=pics[1:],
    duration=200,
    loop=1,
    )


def tune(default_config):
    """
    Hyperparameter sweep with wandb, including logic to:
    - Initialize wandb
    - Train for each hyperparameter set
    - Save checkpoint
    - Evaluate and log GIF
    """
    import copy

    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": "cleanup",
        "method": "grid",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {


            "SEED": {"values": [42, 52, 62]},

        },
    }

    def wrapped_make_train():


        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)

        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v


        run_name = f"sweep_{config['ENV_NAME']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0])


    wandb.login()
    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="config", config_name="lase_harvest_4a")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()
