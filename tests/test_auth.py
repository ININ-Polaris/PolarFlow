def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json == {"status": "ok"}


def test_me_requires_login(client):
    r = client.get("/me")
    # With default Flask-Login behavior this is 401 (API style once unauthorized_handler is set)
    assert r.status_code == 401


def test_login_success_and_me(client, admin_user):
    r = client.post("/auth/login", json={"username": "admin", "password": "secret123"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["user"]["username"] == "admin"

    r2 = client.get("/me")
    assert r2.status_code == 200
    me = r2.get_json()
    assert me["username"] == "admin"
    assert me["visible_gpus"] == [0, 1, 2]


def test_login_failure(client, admin_user):
    r = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_logout(client, admin_user):
    client.post("/auth/login", json={"username": "admin", "password": "secret123"})
    r = client.post("/auth/logout")
    assert r.status_code == 200
    # now /me should be unauthorized again
    r2 = client.get("/me")
    assert r2.status_code == 401


def test_user_loader_handles_bad_cookie(monkeypatch):
    from polar_flow.server.auth import load_user

    assert load_user("not-an-int") is None
