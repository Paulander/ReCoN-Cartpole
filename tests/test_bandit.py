from recon_lite.plasticity import BanditConfig, assign_reward, choose_child, init_bandit_state


def test_bandit_explores_then_prefers_rewarded_arm():
    state = init_bandit_state({"p": ["a", "b"]})
    cfg = BanditConfig(c_explore=0.01)
    assert choose_child("p", state, 1.0, cfg) == "a"
    assign_reward("p", "a", 1.0, state)
    assert choose_child("p", state, 1.0, cfg) == "b"
    assign_reward("p", "b", -1.0, state)
    assert choose_child("p", state, 0.01, cfg) == "a"

