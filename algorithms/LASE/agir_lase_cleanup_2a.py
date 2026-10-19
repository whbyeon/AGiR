"""Based on PureJaxRL and JaxMARL. LASE with Adaptive Gift Refusal (AGiR).
Uses other agents' actual observations instead of an imagination module.
Legacy RNN machinery is retained but is not part of the method."""

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


from typing import Literal, Optional

class WRegressorRNN(nn.Module):
    hidden_dim: int = 64
    rnn_type: Literal["GRU", "LSTM"] = "GRU"
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, joint_action, reward, prev_hidden):
        """Legacy auxiliary component retained for compatibility; not part of the method."""
        T, E = obs.shape[:2]
        act_fn = nn.relu if self.activation == "relu" else nn.tanh


        obs_flat = obs.reshape((T * E, *obs.shape[2:]))


        obs_emb = nn.Conv(features=16, kernel_size=(3,3), padding="SAME")(obs_flat)
        obs_emb = obs_emb.reshape((obs_emb.shape[0], -1))
        obs_emb = act_fn(nn.Dense(features=self.hidden_dim)(obs_emb))

        obs_emb = obs_emb.reshape((T, E, -1))


        x = jnp.concatenate([obs_emb, joint_action, reward], axis=-1)


        x = nn.Dense(self.hidden_dim)(x)
        x = act_fn(x)


        if self.rnn_type == "GRU":
            class GRUWrapper(nn.Module):
                features: int

                @nn.compact
                def __call__(self, carry, x):
                    return nn.GRUCell(self.features)(carry, x)

            scanned_rnn = nn.scan(
                GRUWrapper,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0
            )(features=self.hidden_dim)


            init_carry = prev_hidden

        else:
            class LSTMWrapper(nn.Module):
                features: int

                @nn.compact
                def __call__(self, carry, x):
                    return nn.LSTMCell(self.features)(carry, x)

            scanned_rnn = nn.scan(
                LSTMWrapper,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0
            )(features=self.hidden_dim)


            init_carry = prev_hidden

        last_hidden, rnn_outputs = scanned_rnn(init_carry, x)


        w_pred = nn.Dense(1)(rnn_outputs)
        w_pred = nn.sigmoid(w_pred)
        return w_pred, last_hidden


def compute_w_i_t(update_step, t, i, reward_hist, gift_hist, w_prev, config, rng):
    """
    update_step, t: jnp.int32          # to use for jax cond
    i: int
    reward_hist, gift_hist: (A, window, E)
    w_prev: (A, E)
    """


    R_agg = jnp.sum(reward_hist[i], axis=0)     # (E,)
    G_agg = jnp.sum(gift_hist[i], axis=0)     # (E,)
    G_safe = jnp.where(G_agg == 0, 1e-8, G_agg)

    E = config["NUM_ENVS"]
    rng, _rng = jax.random.split(rng)
    rng_envs = jax.random.split(_rng, config["NUM_ENVS"])

    V = jnp.zeros_like(R_agg)  # (E,)

    # Compare reward-gift orientation with its mean across parallel environments.
    angle = jnp.arctan(R_agg / G_safe)  # (E,)
    avg_angle = jnp.mean(angle)

    # C = 1 in the AGiR step-size modulator.
    delta_scale = 2 * jax.nn.sigmoid(
        (-config["REFUSE_KWARGS"]["beta"] * R_agg
        - (1 - config["REFUSE_KWARGS"]["beta"]) * V) / 1
    )
    delta = (
        config["REFUSE_KWARGS"]["alpha"]
        * w_prev[i]
        * delta_scale
        * (angle - avg_angle)
    )  # (E,)


    w_update = jnp.clip(
        w_prev[i] + delta,
        config["REFUSE_KWARGS"]["w_min"],
        config["REFUSE_KWARGS"]["w_max"],
    )  # (E,)
    w_update = w_update[:, None]  # (E,1)


    # The initial acceptance weight conditions the first action before any reward is observed.

    def init_case(rng_all):
        shape = w_update.shape  # (E,1)
        w_init_mode = config["REFUSE_KWARGS"]["w_init"]
        return jax.lax.switch(

            jnp.array(
                {"all_zero": 0, "all_one": 1, "all_half": 2, "random": 3}[w_init_mode]
            ),
            [
                lambda _: jnp.zeros(shape),
                lambda _: jnp.ones(shape),
                lambda _: jnp.full(shape, 0.5),
                lambda rng_: jax.vmap(lambda rng_one:jax.random.uniform(rng_one))(rng_).reshape((E,1)),
            ],
            rng_all,
        )

    num_sub_epi = config["ENV_KWARGS"]["num_inner_steps"] // config["NUM_STEPS"]
    cond = (update_step % num_sub_epi == 0) & (t == 0)

    w_i_t = jax.lax.cond(cond, init_case, lambda _: w_update, operand=rng_envs)

    return w_i_t    # (E,1)


def window_sum(arr, i, start, end):
    # arr: (A, T_max, E)


    T_max = arr.shape[1]


    hist = arr[i]   # (T_max, E)


    idx = jnp.arange(T_max)
    mask = (idx >= start) & (idx < end)
    masked = hist * mask[:, None]  # (T_max, E)

    return jnp.sum(masked, axis=0)  # (E,)


class ActorCritic_w(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, w):               # x: (batch, obs_dim)
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)


        xw = jnp.concatenate([embedding, w], axis=-1)  # (batch, emb_dim + 1)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(xw)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(xw)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Distill():
    pass


class Imagine_w(nn.Module):

    @nn.compact
    def __call__(self, obs, other_id_onehot, w):  # obs: (B, C, H, W)
        B, C, H, W = obs.shape
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(obs)  # (B, 16, H, W)
        x = x.reshape((x.shape[0], -1))                                    # (B, 16*H*W)
        x = jnp.concatenate([x, other_id_onehot, w], axis=-1)                 # (B, 16*H*W + A + 1)
        x = nn.relu(nn.Dense(features=C * H * W)(x))  # (B, C*H*W)
        x = nn.sigmoid(nn.Dense(features=C * H * W)(x))  # (B, C*H*W)
        x = x.reshape((x.shape[0], C, H, W))  # (B, C, H, W)
        return x


class SRPolicy_w(nn.Module):      # Estimate the best-response policy against the current opponents.
    action_dim: int

    @nn.compact
    def __call__(self, obs, w):  # obs: (B, C, H, W)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(obs)  # (B, 16, H, W)
        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, w], axis=-1)        # (B, 16*H*W + 1)
        x = nn.relu(nn.Dense(features=64)(x))
        logits = nn.Dense(features=self.action_dim)(x)
        return distrax.Categorical(logits=logits)


class SRQNet_w(nn.Module):        # Estimate the best-response Q-function against the current opponents.

    @nn.compact
    def __call__(self, obs, joint_action, w):  # obs: (B, C, H, W)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(obs)
        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, joint_action, w], axis=-1)  # (B, C*H*W + A*action_dim + 1)
        x = nn.relu(nn.Dense(features=64)(x))
        q = nn.Dense(features=1)(x)
        return jnp.squeeze(q, axis=-1)  # (B,)


def compute_gifting_weights_single(i, obs, joint_action,
                                   sr_q_apply, sr_q_params_i,
                                   sr_policy_apply, sr_policy_params_i,
                                   w_i,
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

            pi_sr = sr_policy_apply(sr_policy_params_i, o_hat_j, w_i)
            a_j_probs = pi_sr.probs  # (B, action_dim)

            def integrate_counterfactual(a_j_prime, w_i):
                a_new = a_joint.at[:, j].set(nn.one_hot(a_j_prime, action_dim))  # (B, A, action_dim)
                return sr_q_apply(sr_q_params_i, obs_i, a_new.reshape(B, -1), w_i)  # (B,)

            a_j_vals = jnp.arange(action_dim)
            counterfactual_qs = jax.vmap(integrate_counterfactual, in_axes=(0, None))(a_j_vals, w_i)  # (action_dim, B)
            expected_q = jnp.sum(a_j_probs.T * counterfactual_qs, axis=0)     # (B,)
            q_max = jnp.max(counterfactual_qs, axis=0)
            q_min = jnp.min(counterfactual_qs, axis=0)
            denom = (A - 1) * (q_max - q_min + 1e-8)

            true_q = sr_q_apply(sr_q_params_i, obs_i, joint_action, w_i)  # (B,)
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
    w: jnp.ndarray


def get_rollout(params, config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    network = [ActorCritic_w(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
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
        assert config["ENV_KWARGS"]["num_inner_steps"] % config["NUM_STEPS"] == 0, "num_inner_steps must be divided by NUM_STEPS for correct logging"

        # Initialize policy and value networks.
        network = [ActorCritic_w(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        init_w = jnp.zeros((1, 1))

        network_params = [network[i].init(_rng, init_x, init_w) for i in range(env.num_agents)]


        w_rnn_model = [WRegressorRNN(
                            hidden_dim=64,
                            rnn_type="GRU",
                            activation=config["ACTIVATION"],
                        ) for _ in range(env.num_agents)]
        dummy_obs = jnp.zeros((1,1, *(env.observation_space()[0]).shape))
        dummy_joint_action = jnp.zeros((1,1, env.num_agents))
        dummy_reward = jnp.zeros((1,1, 1))                                            # (1,1,...)
        dummy_rnn_hidden = jnp.zeros((1, 64))
        rnn_params = [w_rnn_model[i].init(_rng, dummy_obs, dummy_joint_action, dummy_reward, dummy_rnn_hidden) for i in range(env.num_agents)]


        # Initialize LASE social-relationship modules.
        dummy_id = jnp.zeros((1, env.num_agents))
        dummy_action = jnp.zeros((1, env.num_agents * env.action_space().n))

        imagine_model = [Imagine_w() for _ in range(env.num_agents)]
        sr_policy_model = [SRPolicy_w(env.action_space().n) for _ in range(env.num_agents)]
        sr_q_model = [SRQNet_w() for _ in range(env.num_agents)]

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        imagine_params = [imagine_model[i].init(rng1, init_x, dummy_id, init_w) for i in range(env.num_agents)]
        sr_policy_params = [sr_policy_model[i].init(rng2, init_x, init_w) for i in range(env.num_agents)]
        sr_q_params = [sr_q_model[i].init(rng3, init_x, dummy_action, init_w) for i in range(env.num_agents)]


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

            rnn_tx = optax.chain(
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
                optax.adam(config["LASE_KWARGS"]["LR_SR"], eps=1e-5),
            )
            im_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LASE_KWARGS"]["LR_IMAGINE"], eps=1e-5),
            )

            rnn_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["RNN_LR"], eps=1e-5),
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

        rnn_train_state = [TrainState.create(
            apply_fn=w_rnn_model[i].apply,
            params=rnn_params[i],
            tx=rnn_tx,
        ) for i in range(env.num_agents)]


        # Initialize parallel environments.
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # Training loop.
        def _update_step(runner_state, unused):


            def _env_step_realtime_refuse(runner_state, unused):
                train_state, sr_q_state, sr_policy_state, im_train_state, rnn_train_state, env_state, last_obs, update_step, rng, reward_hist, gift_hist, w_prev, cum_before_done, rnn_hidden, t = runner_state
                # reward_hist, gift_hist: (A, window, E)

                # Reset at episode boundaries, preserving state across rollouts.
                # w_prev: (A, E)


                # Reset histories at episode boundaries; preserve them across rollouts.
                # (t == 0)

                num_sub_epi = config["ENV_KWARGS"]["num_inner_steps"] // config["NUM_STEPS"]
                cond = (update_step % num_sub_epi == 0) & (t == 0)

                reward_hist = jnp.where(cond, jnp.zeros_like(reward_hist), reward_hist)
                gift_hist = jnp.where(cond, jnp.zeros_like(gift_hist), gift_hist)

                E = reward_hist.shape[2]
                window = reward_hist.shape[1]
                assert E == config["NUM_ENVS"], "error in E"
                assert window == config["REFUSE_KWARGS"]["w_period"], "error in window"


                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))                   # (A, E, obs_dim)

                env_act = {}
                log_prob = []
                value = []
                w_t = []
                for i in range(env.num_agents):
                    print(f"input_obs_shape agent{i}", obs_batch[i].shape)      # (E, obs_dim)


                    w_i_t = compute_w_i_t(update_step, t, i, reward_hist, gift_hist, w_prev, config, _rng)
                    # w_i_t controls gift acceptance, distinct from LASE gifting weights.


                    pi, value_i = network[i].apply(train_state[i].params, obs_batch[i], w_i_t) # (E, action_dim)
                    action = pi.sample(seed=_rng)           # (E,)
                    log_prob.append(pi.log_prob(action))
                    env_act[env.agents[i]] = action         # action: (E,)
                    value.append(value_i)                   # List of arrays, each (E,)
                    w_t.append(w_i_t.squeeze(-1))
                w_t = jnp.stack(w_t)
                env_act = [v for v in env_act.values()]


                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )


                # Compute gifting weights from the observation before the environment step.


                num_envs = config["NUM_ENVS"]
                num_agents = env.num_agents
                action_dim = env.action_space().n

                obs_tensor = last_obs  # (E,A,obs_dim)

                action = jnp.stack(env_act).T         # (E,A)
                action_onehot = jax.nn.one_hot(action, action_dim)  # (E,A,action_dim)
                joint_action = action_onehot.reshape(num_envs, num_agents * action_dim)     # (E,A*action_dim)

                agent_ids_onehot = jnp.eye(num_agents)

                # compute gifting weights: (E, A, A)
                # weights distribute outgoing rewards; w_i controls incoming gift acceptance.
                weights = []
                for i in range(env.num_agents):
                    w_i = jnp.expand_dims(w_t[i], axis=-1)      # (E,1)
                    weights_i = compute_gifting_weights_single(
                        i=i,
                        obs=obs_tensor,
                        joint_action=joint_action,
                        sr_q_apply=sr_q_model[i].apply,
                        sr_q_params_i=sr_q_params[i],
                        sr_policy_apply=sr_policy_model[i].apply,
                        sr_policy_params_i=sr_policy_params[i],
                        w_i=w_i,
                        imagine_apply=imagine_model[i].apply,
                        imagine_params_i=imagine_params[i],
                        agent_ids_onehot=agent_ids_onehot,
                        config={"num_agents": num_agents, "action_dim": action_dim},
                    )
                    weights.append(weights_i)
                weights = jnp.stack(weights, axis=1)    # (E, giver, receiver)


                diag = 1.0 - jnp.sum(weights, axis=-1)  # (E, A)
                weights = weights.at[:, jnp.arange(num_agents), jnp.arange(num_agents)].set(diag)

                reward_tensor = reward   # (E, A)

                # Received gift: sum of incoming transfers from other agents.
                weights_no_diag = weights.at[:, jnp.arange(num_agents), jnp.arange(num_agents)].set(0.0)
                gift_reward = jnp.einsum("eij,ej->ei", weights_no_diag.transpose((0, 2, 1)), reward_tensor)  # (E,A)

                # Given gift: the outgoing fraction of the agent's own reward.
                w_ii = jnp.einsum("eii->ei", weights)  # (E, A)
                gift_given = (1.0 - w_ii) * reward_tensor    # (E, A)


                transition = []
                done = [v for v in done.values()]
                for i in range(env.num_agents):
                    info_i = {key: jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ENVS"],1)), value[:,i]) for key, value in info.items()}

                    transition.append(Transition(
                        done[i],
                        env_act[i],         # (E,)
                        value[i],
                        reward[:,i],        # (E,A)->(E,)
                        log_prob[i],
                        obs_batch[i],
                        info_i,                 # (E,1)
                        gift_reward[:,i],        # (E,)
                        gift_given[:,i],          # (E,)
                        w_t[i][:, None],   # w: (E,1)
                    ))


                    r_t = reward[:,i]         # (E,)
                    g_t = gift_reward[:,i]         # (E,)
                    reward_hist = reward_hist.at[i].set(
                        jnp.roll(reward_hist[i], shift=1, axis=0).at[0].set(r_t)
                    )
                    gift_hist = gift_hist.at[i].set(
                        jnp.roll(gift_hist[i], shift=1, axis=0).at[0].set(g_t)
                    )

                t = t + 1
                runner_state = (train_state, sr_q_state, sr_policy_state, im_train_state, rnn_train_state, env_state, obsv, update_step, rng, reward_hist, gift_hist, w_t, cum_before_done, rnn_hidden, t)
                return runner_state, transition


            runner_state, traj_batch = jax.lax.scan(
                _env_step_realtime_refuse, runner_state, None, config["NUM_STEPS"]
            )

            # Compute GAE.
            train_state, sr_q_state, sr_policy_state, im_train_state, rnn_train_state, env_state, last_obs, update_step, rng, reward_hist, gift_hist, last_w_t, cum_before_done, rnn_hidden, t = runner_state
            last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))    # (A,E,obs_dim)
            last_val = []

            # The final observation is o_T; the stored acceptance weight is w_{T-1}.
            # Compute w_T from the final histories before bootstrapping the value.

            new_w_last = []
            for i in range(env.num_agents):

                w_i_last = compute_w_i_t(0, 1, i, reward_hist, gift_hist, last_w_t, config, rng)
                _, last_val_i = network[i].apply(train_state[i].params, last_obs_batch[i], w_i_last)
                last_val.append(last_val_i)
                new_w_last.append(w_i_last.squeeze())   # (E,1) -> (E,)
            last_val = jnp.stack(last_val, axis=0)  # (A,E)
            new_w_last = jnp.stack(new_w_last, axis=0)      # (A,E)


            T = traj_batch[0].obs.shape[0]
            E = traj_batch[0].obs.shape[1]
            action_dim = env.action_space().n

            joint_action_next = []
            for i in range(env.num_agents):


                w_i_last = jnp.expand_dims(new_w_last[i], axis=-1)  # (E,) -> (E,1)
                pi_next, _ = network[i].apply(train_state[i].params, last_obs_batch[i], w_i_last)    # Policy batch: E environments.
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
                flat_w = traj_batch[i].w.reshape((T*E,1))              # (T,E,1) -> (T*E,1)
                flat_q_sr_now = sr_q_state[i].apply_fn(sr_q_state[i].params, flat_obs, action_onehot, flat_w)  # (T*E,)


                w_i_last = jnp.expand_dims(new_w_last[i], axis=-1)  # (E,) -> (E,1)
                q_sr_last = sr_q_state[i].apply_fn(sr_q_state[i].params, last_obs_batch[i], action_next_onehot, w_i_last)  # (E,)
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


                reward_seq = traj_batch.reward                  # (T,E)
                gift_reward_seq = traj_batch.gift_reward        # (T,E)
                gift_given_seq = traj_batch.gift_given          # (T,E)
                w_seq = traj_batch.w.squeeze(-1)                # (T,E,1) -> (T,E)


                shaped_reward = reward_seq - gift_given_seq + w_seq * gift_reward_seq

                # update traj_batch with shaped_reward
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

                        pi, value = network_used.apply(params, traj_batch.obs, traj_batch.w)
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


            loss_info_all = []
            for i in range(env.num_agents):
                update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                )

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

                w_i = traj_batch[i].w   # (T,E,1)

                # Train the SR Q-network.
                def _sr_q_loss_fn(params):
                    joint_action = jnp.stack([traj_batch[j].action for j in range(num_agents)], axis=-1).reshape((-1, num_agents))    # (T,E,A)->(T*E,A)
                    joint_action_onehot = jax.nn.one_hot(joint_action, action_dim).reshape((T*num_envs, -1))      # (T*E,A,action_dim)->(T*E,A*action_dim)
                    obs_dim = traj_batch[i].obs.shape[2:]
                    flat_obs = traj_batch[i].obs.reshape((T*num_envs, *obs_dim))          # (T*E, obs_dim)
                    flat_w = w_i.reshape((T*num_envs, 1))
                    q_pred = sr_q_state.apply_fn(params, flat_obs, joint_action_onehot, flat_w)        # (T*E,)
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
                    flat_w = w_i.reshape((T*num_envs, 1))
                    pi_sr = sr_policy_state.apply_fn(params, flat_obs, flat_w)          # (T*E, action_dim)
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
                    config={"action_dim": config["action_dim"], "num_agents": env.num_agents, "num_envs": config["NUM_ENVS"], "grad_steps": config["LASE_KWARGS"]["GRAD_STEPS"]},
                    i=i,
                )
                sr_q_state[i] = sr_q_state_i
                sr_policy_state[i] = sr_policy_state_i
                sr_loss.append(sr_loss_i)


            sr_loss = jnp.mean(jnp.stack(sr_loss))


            def update_rnn(rnn_train_state, w_rnn_model, traj_batch, prev_rnn_hidden):
                """Legacy auxiliary component retained for compatibility; not part of the method."""
                new_rnn_train_state = []
                losses = []
                for i in range(len(traj_batch)):
                    obs = traj_batch[i].obs
                    reward = traj_batch[i].reward[..., None]
                    w_target = traj_batch[i].w


                    joint_action = jnp.stack([traj_batch[j].action for j in range(len(traj_batch))], axis=-1)

                    num_sub_epi = config["ENV_KWARGS"]["num_inner_steps"] // config["NUM_STEPS"]
                    epi_start = update_step % num_sub_epi == 0

                    def loss_fn(params):
                        init_carry = jnp.where(epi_start, jnp.zeros((config["NUM_ENVS"],64)), prev_rnn_hidden)
                        w_pred, last_hidden = w_rnn_model[i].apply(params, obs, joint_action, reward, init_carry)

                        w_target_cut = w_target[1:, :, :]
                        w_pred_cut   = w_pred[1:, :, :]
                        mse = jnp.mean((w_target_cut - w_pred_cut) ** 2)
                        return mse, last_hidden


                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (loss, last_hidden), grads = grad_fn(rnn_train_state[i].params)
                    new_state = rnn_train_state[i].apply_gradients(grads=grads)

                    new_rnn_train_state.append(new_state)
                    losses.append(loss)

                return new_rnn_train_state, jnp.mean(jnp.stack(losses)), last_hidden

            def rnn_update_branch(args):
                rnn_train_state, traj_batch, rnn_hidden = args
                rnn_train_state, avg_rnn_loss, last_hidden = update_rnn(
                    rnn_train_state, w_rnn_model, traj_batch, rnn_hidden
                )
                metric = {}
                metric["average rnn loss"] = avg_rnn_loss
                return rnn_train_state, metric, last_hidden

            def rnn_no_update_branch(args):
                rnn_train_state, _, rnn_hidden = args
                metric = {}
                metric["average rnn loss"] = 0.0  # Both JAX conditional branches must return the same metric keys.
                return rnn_train_state, metric, rnn_hidden

            num_sub_epi = config["ENV_KWARGS"]["num_inner_steps"] // config["NUM_STEPS"]
            cond1 = update_step >= config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"]) - config["REFUSE_KWARGS"]["num_epi_for_rnn_train"] * num_sub_epi
            cond2 = (update_step % num_sub_epi) == 0
            cond = jnp.logical_and(cond1, cond2)

            rnn_train_state, metric, next_rnn_hidden = jax.lax.cond(
                cond,
                rnn_update_branch,
                rnn_no_update_branch,
                (rnn_train_state, traj_batch, rnn_hidden)
            )


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


            cleaned_water_save = []
            for i in range(env.num_agents):
                cleaned_water_save.append(metric[f"agent{i}_cleaned_water"][:,:1,0])


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


                assert traj_batch[i].done[-1,0].ndim==0, "check the shape of done again!"
                epi_end = traj_batch[i].done[-1,0]

                reward_from_traj_batch = traj_batch[i].reward    # (T,E)
                episode_returns_per_episode_using_reward = jnp.sum(reward_from_traj_batch, axis=0)                    # (E,)
                episode_returns_per_agent_using_reward = jnp.mean(episode_returns_per_episode_using_reward)

                cum_before_done["ext_reward_carry"] = cum_before_done["ext_reward_carry"].at[i].set(cum_before_done["ext_reward_carry"][i] + episode_returns_per_agent_using_reward)
                cum_before_done["ext_reward"] = cum_before_done["ext_reward"].at[i].set(jnp.where(epi_end,cum_before_done["ext_reward_carry"][i],cum_before_done["ext_reward"][i]))
                metric[f"agent{i}_received_reward_return"] = cum_before_done["ext_reward"][i]      # env is already giving "shaped" reward
                returns_using_reward_average = returns_using_reward_average + cum_before_done["ext_reward"][i]
                cum_before_done["ext_reward_carry"] = cum_before_done["ext_reward_carry"].at[i].set(jnp.where(epi_end,0,cum_before_done["ext_reward_carry"][i]))

                gift_reward = traj_batch[i].gift_reward  # (T,E)
                gift_return_per_episode = jnp.sum(gift_reward, axis=0)  # shape: (E,)
                gift_return_mean_per_agent = jnp.mean(gift_return_per_episode)

                cum_before_done["received_carry"] = cum_before_done["received_carry"].at[i].set(cum_before_done["received_carry"][i] + gift_return_mean_per_agent)
                cum_before_done["received"] = cum_before_done["received"].at[i].set(jnp.where(epi_end,cum_before_done["received_carry"][i],cum_before_done["received"][i]))
                metric[f"agent{i}_received_gift_return"] = cum_before_done["received"][i]
                cum_before_done["received_carry"] = cum_before_done["received_carry"].at[i].set(jnp.where(epi_end,0,cum_before_done["received_carry"][i]))

                gift_reward = traj_batch[i].gift_reward  # (T,E)
                refuse_w = traj_batch[i].w.squeeze(-1)  # (T,E)
                actual_gift_return_per_episode = jnp.sum(gift_reward*refuse_w, axis=0)  # shape: (E,)
                actual_gift_return_mean_per_agent = jnp.mean(actual_gift_return_per_episode)

                cum_before_done["actual_received_carry"] = cum_before_done["actual_received_carry"].at[i].set(cum_before_done["actual_received_carry"][i] + actual_gift_return_mean_per_agent)
                cum_before_done["actual_received"] = cum_before_done["actual_received"].at[i].set(jnp.where(epi_end,cum_before_done["actual_received_carry"][i],cum_before_done["actual_received"][i]))
                metric[f"agent{i}_actual_received_gift_return"] = cum_before_done["actual_received"][i]
                cum_before_done["actual_received_carry"] = cum_before_done["actual_received_carry"].at[i].set(jnp.where(epi_end,0,cum_before_done["actual_received_carry"][i]))

                gift_given = traj_batch[i].gift_given    # shape (T, E)
                gift_given_per_episode = jnp.sum(gift_given, axis=0)  # shape: (E,)
                gift_given_mean_per_agent = jnp.mean(gift_given_per_episode)

                cum_before_done["given_carry"] = cum_before_done["given_carry"].at[i].set(cum_before_done["given_carry"][i] + gift_given_mean_per_agent)
                cum_before_done["given"] = cum_before_done["given"].at[i].set(jnp.where(epi_end,cum_before_done["given_carry"][i],cum_before_done["given"][i]))
                metric[f"agent{i}_given_gift_return"] = cum_before_done["given"][i]
                cum_before_done["given_carry"] = cum_before_done["given_carry"].at[i].set(jnp.where(epi_end,0,cum_before_done["given_carry"][i]))

                metric[f"agent{i}_total_return"] = (
                    returned_episode_returns_per_agent + cum_before_done["received"][i] - cum_before_done["given"][i]
                )
                metric[f"agent{i}_total_return_using_reward"] = (
                    cum_before_done["ext_reward"][i] + cum_before_done["received"][i] - cum_before_done["given"][i]
                )
                metric[f"agent{i}_actual_total_return"] = (
                    cum_before_done["ext_reward"][i] + cum_before_done["actual_received"][i] - cum_before_done["given"][i]
                )


                metric[f"agent{i}_w_env0_time-avg"] = jnp.mean(traj_batch[i].w[:,0,0])

            metric["average_env_return"] = returned_episode_returns_average/env.num_agents
            metric["average_env_return_using_reward"] = returns_using_reward_average/env.num_agents


            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            for i in range(env.num_agents):
                metric[f"agent{i}_clean_action_info"] = metric[f"agent{i}_clean_action_info"] * config["ENV_KWARGS"]["num_inner_steps"]
            jax.debug.callback(callback, metric)


            last_extrinsic_reward_env0 = []
            last_clean_action_env0 = []
            last_w_rollout_env0 = []
            for i in range(env.num_agents):
                last_extrinsic_reward_env0.append(traj_batch[i].reward[:,:1])     # (T, 1)
                last_clean_action_env0.append(jnp.where(traj_batch[i].action[:,:1]==8, 1, 0)) # (T, 1)

                last_w_rollout_env0.append(traj_batch[i].w[:,:1,0])       # (T,E_to_save=1)
            metric["last_extrinsic_reward_env0"] = jnp.stack(last_extrinsic_reward_env0, axis=0)  # (A,T,E=1)

            metric["last_clean_action_env0"] = jnp.stack(last_clean_action_env0, axis=0)  # (A, T, 1)
            metric["last_cleaned_water_env0"] = jnp.stack(cleaned_water_save, axis=0)   # (A, T, 1)
            metric["last_w_rollout_env0"] = jnp.stack(last_w_rollout_env0, axis=0)            # (A,T,E=1)


            # Carry w_T into the next rollout.

            t = jnp.where(epi_end, jnp.array(0, dtype=jnp.int32), t)

            runner_state = (train_state, sr_q_state, sr_policy_state, im_train_state, rnn_train_state, env_state, last_obs, update_step, rng, reward_hist, gift_hist, new_w_last, cum_before_done, next_rnn_hidden, t)
            return runner_state, metric


        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, sr_q_state, sr_policy_state, im_train_state, rnn_train_state, env_state, obsv, jnp.array(0, dtype=jnp.int32), _rng, jnp.zeros((env.num_agents, config["REFUSE_KWARGS"]["w_period"], config["NUM_ENVS"])), jnp.zeros((env.num_agents, config["REFUSE_KWARGS"]["w_period"], config["NUM_ENVS"])), jnp.ones((env.num_agents, config["NUM_ENVS"])),
                        {"ext_reward": jnp.zeros(env.num_agents), "received": jnp.zeros(env.num_agents),"given": jnp.zeros(env.num_agents), "actual_received": jnp.zeros(env.num_agents),
                         "ext_reward_carry": jnp.zeros(env.num_agents), "received_carry": jnp.zeros(env.num_agents),"given_carry": jnp.zeros(env.num_agents), "actual_received_carry": jnp.zeros(env.num_agents)}, jnp.zeros((config["NUM_ENVS"], 64)), jnp.array(0, dtype=jnp.int32))
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

def single_run(config):
    config = OmegaConf.to_container(config)


    wandb.login()

    group_name = f'env=cleanup_2a_{config["MAP_NAME"]}_algo={config["REFUSE_KWARGS"]["refusing_method"]}-LASE_winit={config["REFUSE_KWARGS"]["w_init"]}_H={config["REFUSE_KWARGS"]["w_period"]}_nb={config["NUM_MINIBATCHES"]}_lr={config["LR"]}_env={config["NUM_ENVS"]}_ns={config["NUM_STEPS"]}_es={config["ENV_KWARGS"]["num_inner_steps"]}_gs={config["LASE_KWARGS"]["GRAD_STEPS"]}'
    run_name=f'seed={config["SEED"]} cleanup_2a_{config["MAP_NAME"]} H={config["REFUSE_KWARGS"]["w_period"]} scrw-lase {config["REFUSE_KWARGS"]["w_init"]},rnn_epi={config["REFUSE_KWARGS"]["num_epi_for_rnn_train"]}_beta={config["REFUSE_KWARGS"]["beta"]},{config["REFUSE_KWARGS"]["w_min"]},{config["REFUSE_KWARGS"]["w_max"]},gradsteps={config["LASE_KWARGS"]["GRAD_STEPS"]} {config["NUM_ENVS"]},{config["NUM_STEPS"]},{config["ENV_KWARGS"]["num_inner_steps"]}'
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=run_name,
        group=group_name,
        job_type="cleanup",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'lase-{config["REFUSE_KWARGS"]["refusing_method"]}_winit_{config["REFUSE_KWARGS"]["w_init"]}_re{config["REFUSE_KWARGS"]["num_epi_for_rnn_train"]}_beta{config["REFUSE_KWARGS"]["beta"]}_{config["MAP_NAME"]}_seed{config["SEED"]}'


    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    rnn_train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][4])


    import pandas as pd
    excel_dir = f'last_w_r_clean/cleanup_{config["MAP_NAME"]}_{config["ENV_KWARGS"]["num_agents"]}a'
    path = Path(excel_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename_excel = f'{config["MAP_NAME"]}_LASE_{config["REFUSE_KWARGS"]["refusing_method"]}_winit_{config["REFUSE_KWARGS"]["w_init"]}_re{config["REFUSE_KWARGS"]["num_epi_for_rnn_train"]}_beta{config["REFUSE_KWARGS"]["beta"]}_{config["SEED"]}.xlsx'
    filepath = path / filename_excel

    def save_4arrays_to_excel(array1, array2, array3, array4, filename=str(filepath)):
        with pd.ExcelWriter(filename) as writer:
            for name, array in zip(
                ["last_w", "last_extrinsic_reward", "last_clean_action", "last_cleaned_water"],
                [array1, array2, array3, array4]
            ):
                # shape: (A, T, E)
                reshaped = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
                df = pd.DataFrame(reshaped)
                df.to_excel(writer, sheet_name=name, index=False)

        print(f"Saved to {filename}")


    A = config["ENV_KWARGS"]["num_agents"]
    T = config["NUM_STEPS"]
    num_sub_epi = config["ENV_KWARGS"]["num_inner_steps"] // config["NUM_STEPS"]
    # (num_sub_epi, A, T, E) -> (A, num_sub_epi, T, E) -> (A, num_sub_epi*T, E)
    w_array = jnp.transpose(out["metrics"]["last_w_rollout_env0"][0, -num_sub_epi:, :, :, :], (1,0,2,3)).reshape((A, num_sub_epi*config["NUM_STEPS"],-1))
    r_array = jnp.transpose(out["metrics"]["last_extrinsic_reward_env0"][0, -num_sub_epi:, :, :, :], (1,0,2,3)).reshape((A, num_sub_epi*config["NUM_STEPS"],-1))
    clean_action_array = jnp.transpose(out["metrics"]["last_clean_action_env0"][0, -num_sub_epi:, :, :, :], (1,0,2,3)).reshape((A, num_sub_epi*config["NUM_STEPS"],-1))
    cleaned_water_array = jnp.transpose(out["metrics"]["last_cleaned_water_env0"][0, -num_sub_epi:, :, :, :], (1,0,2,3)).reshape((A, num_sub_epi*config["NUM_STEPS"],-1))
    save_4arrays_to_excel(
        w_array,
        r_array,
        clean_action_array,
        cleaned_water_array
    )


    last_w_env0 = out["metrics"]["last_w_rollout_env0"][0, -num_sub_epi:, :, :, 0]  # (num_sub_epi,A,T)
    last_w_env0 = jnp.transpose(last_w_env0, (1,0,2)).reshape((A, num_sub_epi*T))   # (A,num_sub_epi*T)

    print(f'shape of last_extrinsic_reward before: {out["metrics"]["last_extrinsic_reward_env0"].shape}')
    last_extrinsic_reward_env0 = out["metrics"]["last_extrinsic_reward_env0"][0, -num_sub_epi:, :, :, 0]   # (1,num_updates,A,T,E) -> (num_sub_epi,A,T)
    last_extrinsic_reward_env0 = jnp.transpose(last_extrinsic_reward_env0, (1,0,2)).reshape((A, num_sub_epi*T))  # (num_sub_epi,A,T) -> (A,num_sub_epi,T) -> (A,num_sub_epi*T)
    print(f"shape of last_extrinsic_reward: {last_extrinsic_reward_env0.shape}")   # (A,num_sub_epi*T)


    last_clean_action_env0 = out["metrics"]["last_clean_action_env0"][0, -num_sub_epi:, :, :, 0]         # (1,num_updates,A,T,E) -> (num_sub_epi,A,T)
    last_clean_action_env0 = jnp.transpose(last_clean_action_env0, (1,0,2)).reshape((A, num_sub_epi*T))   # (num_sub_epi,A,T) -> (A,num_sub_epi,T) -> (A,num_sub_epi*T)


    last_cleaned_water_env0 = out["metrics"]["last_cleaned_water_env0"][0, -num_sub_epi:, :, :, 0]         # (1,num_updates,A,T,E) -> (num_sub_epi,A,T)
    last_cleaned_water_env0 = jnp.transpose(last_cleaned_water_env0, (1,0,2)).reshape((A, num_sub_epi*T))   # (num_sub_epi,A,T) -> (A,num_sub_epi,T) -> (A,num_sub_epi*T)


    for i in range(A):
        wandb.define_metric(f"last_w_env0_agent{i}", step_metric="last_w_step")
        wandb.define_metric(f"last_ext_reward_env0_agent{i}", step_metric="last_w_step")
        wandb.define_metric(f"last_clean_action_env0_agent{i}", step_metric="last_w_step")
        wandb.define_metric(f"last_cleaned_water_env0_agent{i}", step_metric="last_w_step")

    for t in range(T):
        log_dict = {"last_w_step": t}
        for i in range(A):
            log_dict[f"last_w_env0_agent{i}"] = last_w_env0[i, t]
            log_dict[f"last_ext_reward_env0_agent{i}"] = last_extrinsic_reward_env0[i, t]
            log_dict[f"last_clean_action_env0_agent{i}"] = last_clean_action_env0[i, t]
            log_dict[f"last_cleaned_water_env0_agent{i}"] = last_cleaned_water_env0[i, t]
        wandb.log(log_dict)


    params = []
    rnn_params = []
    for i in range(config['ENV_KWARGS']['num_agents']):
        save_path = f"./checkpoints/individual/{filename}_{i}.pkl"
        save_params(train_state[i], save_path)
        params.append(load_params(save_path))

        save_path = f"./checkpoints/individual/{filename}_rnn_{i}.pkl"
        save_params(rnn_train_state[i], save_path)
        rnn_params.append(load_params(save_path))

    evaluate(params, rnn_params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)


def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)

def evaluate(params, rnn_params, env, save_path, config):

    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f'evaluation/cleanup_4a_{config["MAP_NAME"]}'
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)


    env_act = {}
    history_list = [None for _ in range(env.num_agents)]


    for o_t in range(config["GIF_NUM_FRAMES"]):

        obs_batch = jnp.stack([obs[a] for a in env.agents])     # (A, obs_dim)
        A = obs_batch.shape[0]
        obs_dim = obs_batch.shape[1:]

        network = [ActorCritic_w(action_dim=env.action_space().n, activation="relu") for _ in range(env.num_agents)]
        w_regressor_rnn = [WRegressorRNN(hidden_dim=64,rnn_type="GRU",activation=config["ACTIVATION"]) for _ in range(env.num_agents)]

        w_list = []
        for i in range(env.num_agents):
            obs = jnp.expand_dims(obs_batch[i],axis=0)      # (1, obs_dim)


            if o_t == 0:
                w = {
                    "all_zero": jnp.zeros((1,1)),
                    "all_one":  jnp.ones((1,1)),
                    "all_half": jnp.full((1,1), 0.5),
                    "random":   jax.random.uniform(_rng, (1,1)),
                }[config["REFUSE_KWARGS"]["w_init"]]
                history_list[i] = jnp.zeros((1, 64))
            else:
                last_action = jnp.array([v.item() for v in env_act.values()]).reshape((1,1,env.num_agents))
                last_reward = reward[i].reshape((1,1,1))

                w, history_i = w_regressor_rnn[i].apply(
                    rnn_params[i],
                    jnp.expand_dims(obs, axis=0),   # (1,1, obs_dim)
                    last_action,                    # (1,1, A)
                    last_reward,                    # (1,1, 1)
                    history_list[i]                         # (1, 64)
                )
                w = w.squeeze(0)   # (1,1)
                history_list[i] = history_i

            pi, _ = network[i].apply(params[i], obs, w)   # (1, obs_dim)
            rng, _rng = jax.random.split(rng)
            single_action = pi.sample(seed=_rng)
            env_act[env.agents[i]] = single_action

            w_list.append(w.item())


        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]


        img = env.render(state)
        pics.append(img)


        log_dict = {"eval_step": o_t}


        actions = [v.item() for v in env_act.values()]


        rewards = reward.tolist()


        for i in range(env.num_agents):
            log_dict[f"agent{i}_eval_action"] = actions[i]
            log_dict[f"agent{i}_eval_reward"] = rewards[i]
            log_dict[f"agent{i}_eval_w"] = w_list[i]

        wandb.log(log_dict)

        print(f'eval iter {o_t}')


    from datetime import datetime
    import pytz


    korea_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(korea_tz)

    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

    gif_path = f'{root_dir}/algo={config["REFUSE_KWARGS"]["refusing_method"]}-LASE_{formatted_time}_frame{o_t+1}_{config["MAP_NAME"]}_winit_{config["REFUSE_KWARGS"]["w_init"]}_re{config["REFUSE_KWARGS"]["num_epi_for_rnn_train"]}_b={config["REFUSE_KWARGS"]["beta"]}_nb={config["NUM_MINIBATCHES"]}_ns={config["NUM_STEPS"]}_seed={config["SEED"]}.gif'


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


@hydra.main(version_base=None, config_path="config", config_name="agir_lase_cleanup_2a")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()
