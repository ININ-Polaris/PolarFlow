use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use dotenvy::dotenv;
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use pam::Authenticator;
use serde::{Deserialize, Serialize};
use std::{env, fs, net::SocketAddr, sync::Arc};
use thiserror::Error;
use time::{Duration, OffsetDateTime};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

/// 应用全局状态，集中管理配置与密钥
#[derive(Clone)]
struct AppState {
    jwt_key: Arc<EncodingKey>, // HS256 对称密钥（从文件读取）
    jwt_exp_minutes: i64,
    pam_service: String,
}

/// /auth/token 请求体
#[derive(Deserialize)]
struct TokenRequest {
    username: String,
    password: String,
}

/// /auth/token 响应体
#[derive(Serialize)]
struct TokenResponse {
    access_token: String,
    token_type: &'static str,
    expires_in: i64,
}

/// 统一错误响应结构
#[derive(Serialize)]
struct ErrorMessage {
    error: String,
}

/// JWT Claims，尽量最小化；包含必要字段与可选 uid/gid
#[derive(Serialize)]
struct Claims {
    // Slurm 要求的用户字段（默认字段名为 "sun"）
    sun: String,
    iat: i64,
    exp: i64,
}

/// API 错误类型
#[derive(Error, Debug)]
enum ApiError {
    #[error("bad request")]
    BadRequest(&'static str),
    #[error("authentication failed")]
    AuthFailed,
    #[error("internal error")]
    Internal(&'static str),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        // 默认模糊错误，避免泄露认证细节
        let (status, msg) = match self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, m.to_string()),
            ApiError::AuthFailed => (
                StatusCode::UNAUTHORIZED,
                "authentication failed".to_string(),
            ),
            ApiError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal error".to_string(),
            ),
        };
        (status, Json(ErrorMessage { error: msg })).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 读取 .env（开发友好；生产建议用 systemd EnvironmentFile/密钥管理）
    dotenv().ok();

    // 初始化日志：RUST_LOG=info ./pam-jwt-issuer
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // 读取 HS256 密钥文件路径
    let key_path = env::var("JWT_KEY_PATH").unwrap_or_else(|_| "jwt_hs256.key".to_string());
    // 读取并修剪换行
    let secret_bytes = fs::read(&key_path).map_err(|e| {
        error!("failed to read key file {}: {}", key_path, e);
        e
    })?;

    if secret_bytes.len() < 32 {
        error!("JWT secret seems too short; please use at least 32 random bytes for HS256");
    }

    // 构造应用状态
    let state = AppState {
        jwt_key: Arc::new(EncodingKey::from_secret(&secret_bytes)),
        jwt_exp_minutes: env::var("JWT_EXPIRE_MINUTES")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(60),
        pam_service: env::var("PAM_SERVICE").unwrap_or_else(|_| "flaskapi".into()),
    };

    // 路由：
    // - GET /healthz 用于探活
    // - POST /auth/token 签发 JWT（唯一业务功能）
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/auth/token", post(issue_token))
        .with_state(state);

    // 监听地址
    let addr: SocketAddr = env::var("BIND_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".into())
        .parse()?;

    info!("listening on http://{addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

/// 处理签发逻辑：PAM 认证成功后生成 JWT
async fn issue_token(
    State(state): State<AppState>,
    Json(req): Json<TokenRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let username = req.username.trim();
    let password = req.password;

    if username.is_empty() || password.is_empty() {
        return Err(ApiError::BadRequest("username and password are required"));
    }

    // —— PAM 认证开始 ——
    // 使用 pam crate 提供的 Authenticator，并指定 PAM 服务名（对应 /etc/pam.d/<service>）
    let mut auth = Authenticator::with_password(&state.pam_service)
        .map_err(|_| ApiError::Internal("pam init"))?;
    // 设置用户名与口令
    auth.get_handler().set_credentials(username, &password);
    // 调用认证
    auth.authenticate().map_err(|_| ApiError::AuthFailed)?;
    // 如需打开/关闭会话，可使用 open_session/close_session，这里不需要。

    // —— 生成 JWT ——
    let now = OffsetDateTime::now_utc();
    let exp = now + Duration::minutes(state.jwt_exp_minutes);

    let claims = Claims {
        sun: username.to_string(),
        iat: now.unix_timestamp(), // 签发时间
        exp: exp.unix_timestamp(), // 过期时间
    };

    let header = Header::new(Algorithm::HS256);

    let token =
        encode(&header, &claims, &state.jwt_key).map_err(|_| ApiError::Internal("jwt encode"))?;

    Ok((
        StatusCode::OK,
        Json(TokenResponse {
            access_token: token,
            token_type: "Bearer",
            expires_in: state.jwt_exp_minutes * 60,
        }),
    ))
}
