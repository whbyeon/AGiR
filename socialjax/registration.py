from socialjax.environments import (

    Harvest_open_unfair,
    Clean_up_4a_Alow,
    Clean_up_4a_Amid,
    Clean_up_2a_competence_gap_r025,
    Clean_up_2a_mismatch,
    Clean_up_2a_competence_gap_r01,
    Clean_up_2a_no_gap,
    Clean_up_2a_spawn_gap,
    CoinGame,
)


REGISTERED_ENVS = [

    "coin_game",
    "harvest_unfair",
    "clean_up_4a_Alow",
    "clean_up_4a_Amid",
    "clean_up_2a_competence_gap_r025",
    "clean_up_2a_mismatch",
    "clean_up_2a_competence_gap_r01",
    "clean_up_2a_no_gap",
    "clean_up_2a_spawn_gap",
]


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in REGISTERED_ENVS:
        raise ValueError(f"{env_id} is not in registered SocialJax environments")
    elif env_id == "harvest_unfair":
        env = Harvest_open_unfair(**env_kwargs)
    elif env_id == "clean_up_4a_Alow":
        env = Clean_up_4a_Alow(**env_kwargs)
    elif env_id == "clean_up_4a_Amid":
        env = Clean_up_4a_Amid(**env_kwargs)
    elif env_id == "clean_up_2a_competence_gap_r025":
        env = Clean_up_2a_competence_gap_r025(**env_kwargs)
    elif env_id == "clean_up_2a_mismatch":
        env = Clean_up_2a_mismatch(**env_kwargs)
    elif env_id == "clean_up_2a_competence_gap_r01":
        env = Clean_up_2a_competence_gap_r01(**env_kwargs)
    elif env_id == "clean_up_2a_no_gap":
        env = Clean_up_2a_no_gap(**env_kwargs)
    elif env_id == "clean_up_2a_spawn_gap":
        env = Clean_up_2a_spawn_gap(**env_kwargs)
    elif env_id == "coin_game":
        env = CoinGame(**env_kwargs)
    return env
