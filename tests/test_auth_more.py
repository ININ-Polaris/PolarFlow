from polar_flow.server.auth import admin_required


def test_login_missing_fields_returns_400(client):
    r = client.post("/auth/login", json={"username": "x"})  # 没有 password
    assert r.status_code == 400
    assert "required" in r.get_json().get("error", "")


def test_admin_required_unauth(app):
    # 动态注册一个受 admin_required 保护的路由，未登录 → 401（触发 unauthorized_handler）
    @app.route("/_admin_only")
    @admin_required
    def _admin_only():
        return "ok"

    with app.test_client() as c:
        r = c.get("/_admin_only")
        assert r.status_code == 401


def test_admin_required_forbidden(app, normal_user):
    with app.app_context():

        @app.route("/_admin_only2")
        @admin_required
        def _admin_only2():
            return "ok2"

    with app.test_client() as c:
        c.post("/auth/login", json={"username": "alice", "password": "secret123"})
        r = c.get("/_admin_only2")
        assert r.status_code == 403
