from pathlib import Path

from envs.bellman_ford_env import BellmanFordEnv
from gnarl_transfer.expert_data import collect_bf_expert_data, load_expert_dataset, save_expert_dataset


def make_env():
    return BellmanFordEnv(
        n_nodes=4,
        max_nodes=4,
        train_nodes=[4],
        fixed_nodes=4,
        reward_mode="sparse",
        seed=123,
        use_clrs=False,
    )


def test_collect_and_reload_expert_dataset(tmp_path: Path):
    dataset = collect_bf_expert_data(make_env, episodes=3, seed=11)

    assert len(dataset) > 0
    assert set(dataset.observations.keys()) == {"d", "pred", "visited", "source", "t", "p"}
    assert dataset.actions.shape[0] == len(dataset)
    assert dataset.episode_ids.shape[0] == len(dataset)

    path = tmp_path / "expert_dataset.npz"
    save_expert_dataset(path, dataset)
    reloaded = load_expert_dataset(path)

    assert reloaded.actions.shape == dataset.actions.shape
    assert reloaded.episode_ids.shape == dataset.episode_ids.shape
    for key in dataset.observations:
        assert reloaded.observations[key].shape == dataset.observations[key].shape