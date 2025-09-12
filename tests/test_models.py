from polar_flow.server.models import Role, User


def test_password_hashing_is_effective():
    u = User(username="t", role=Role.USER, visible_gpus=[], priority=100)
    u.set_password("abc12345")
    assert u.check_password("abc12345") is True
    assert u.check_password("nope") is False


def test_visible_gpus_default_is_distinct_lists():
    u1 = User(username="u1", role=Role.USER, visible_gpus=[], priority=100)
    u2 = User(username="u2", role=Role.USER, visible_gpus=[], priority=100)
    u1.visible_gpus.append(7)
    assert u1.visible_gpus == [7]
    assert u2.visible_gpus == []
