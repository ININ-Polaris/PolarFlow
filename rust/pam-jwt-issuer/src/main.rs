use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use dotenvy::dotenv;
use jsonwebtoken::{Algorithm, EncodingKey, Header, decode_header, encode};
use pam::Authenticator;
use rsa::{
    Oaep, RsaPrivateKey,
    pkcs1::DecodeRsaPrivateKey,
    pkcs8::{DecodePrivateKey, EncodePublicKey, LineEnding},
};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::{env, fs, io::Write, net::SocketAddr, process::Command, sync::Arc};
use thiserror::Error;
use time::macros::format_description;
use time::{Duration, OffsetDateTime};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

// base64 新 API
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};

// 颁发 SSH 证书所需
use rand::{RngCore, rngs::OsRng};
use tempfile::NamedTempFile;

/// 应用全局状态：JWT 用 HS256，对称密钥；RSA 仅用于传输层解密与导出公钥
#[derive(Clone)]
struct AppState {
    // HS256 对称签名密钥
    jwt_key: Arc<EncodingKey>,
    // RSA 私钥（仅用于 OAEP 解密 & 导出公钥）
    rsa_private: Arc<RsaPrivateKey>,

    jwt_exp_minutes: i64,
    pam_service: String,
    // 抗重放窗口（秒）
    max_payload_age_secs: i64,
}

/// /auth/token 请求体（密文，Base64）
#[derive(Deserialize)]
struct EncryptedTokenRequest {
    ciphertext_b64: String,
}

/// 客户端加密前的明文结构
#[derive(Deserialize)]
struct AuthPayload {
    username: String,
    password: String,
    ts: i64, // unix 秒，用于抗重放
    ssh_pubkey: Option<String>,
}

/// /auth/token 响应体
#[derive(Serialize)]
struct TokenResponse {
    access_token: String,
    token_type: &'static str,
    expires_in: i64,
    ssh_user_cert: Option<String>,  // OpenSSH 用户证书（文本，一行）
    ssh_user_cert_exp: Option<i64>, // 证书到期时间（unix 秒）
}

/// 统一错误响应结构
#[derive(Serialize)]
struct ErrorMessage {
    error: String,
}

/// JWT Claims
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

/// 将 OffsetDateTime 格式化为 ssh-keygen 需要的 UTC "YYYYMMDDHHMMSS"
fn fmt_utc(ts: OffsetDateTime) -> String {
    // OpenSSH 要求无分隔紧凑格式
    // 统一转为 UTC
    let ts_utc = ts.to_offset(time::UtcOffset::UTC);
    ts_utc
        .format(&format_description!(
            "[year][month][day][hour][minute][second]"
        ))
        .unwrap()
}

/// 调用 ssh-keygen 使用 CA 私钥签发用户证书，返回证书文本（*-cert.pub 一行）
fn issue_ssh_user_cert(
    ca_key_path: &str,
    principal: &str,
    valid_after: OffsetDateTime,
    valid_before: OffsetDateTime,
    ssh_pubkey: &str,
    extensions: &[&str],
    serial_base: u64,
) -> Result<String, ApiError> {
    // 1) 用户公钥写入临时文件
    let tmp_dir = env::var("SSH_CERT_TMP_DIR").unwrap_or_else(|_| "/tmp".into());
    let mut pub_tmp = NamedTempFile::new_in(tmp_dir).map_err(|_| ApiError::Internal("tmpfile"))?;
    pub_tmp
        .write_all(ssh_pubkey.as_bytes())
        .map_err(|_| ApiError::Internal("tmp write"))?;
    pub_tmp.flush().ok();

    let pub_path = pub_tmp.path().to_owned();
    let cert_path = pub_path.with_file_name(format!(
        "{}-cert.pub",
        pub_path.file_name().unwrap().to_string_lossy()
    ));

    // 2) 随机序列号（基于 serial_base 做偏移）
    let mut rnd = [0u8; 8];
    OsRng.fill_bytes(&mut rnd);
    let serial = serial_base.wrapping_add(u64::from_le_bytes(rnd));

    // 3) 组织 ssh-keygen 参数
    let from = fmt_utc(valid_after);
    let to = fmt_utc(valid_before);
    let key_id = format!(
        "pam-jwt-issuer:{}:{}",
        principal,
        valid_after.unix_timestamp()
    );

    let mut cmd = Command::new("ssh-keygen");
    cmd.arg("-s")
        .arg(ca_key_path)
        .arg("-I")
        .arg(&key_id)
        .arg("-n")
        .arg(principal)
        .arg("-V")
        .arg(format!("{}:{}", from, to))
        .arg("-z")
        .arg(serial.to_string())
        .arg(&pub_path);

    for ext in extensions {
        if !ext.is_empty() {
            cmd.arg("-O").arg(ext);
        }
    }

    let output = cmd
        .output()
        .map_err(|_| ApiError::Internal("ssh-keygen spawn"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("ssh-keygen failed: {}", stderr);
        return Err(ApiError::Internal("ssh-keygen failed"));
    }

    // 4) 读取生成的证书文本
    let cert_text = fs::read_to_string(&cert_path).map_err(|_| ApiError::Internal("read cert"))?;
    Ok(cert_text.trim().to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 读取 .env
    dotenv().ok();

    // 初始化日志：RUST_LOG=info ./pam-jwt-issuer
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // 读取 HS256 密钥文件路径
    let key_path = env::var("JWT_KEY_PATH").unwrap_or_else(|_| "jwt_hs256.key".to_string());
    let secret_bytes = fs::read(&key_path).map_err(|e| {
        error!("failed to read key file {}: {}", key_path, e);
        e
    })?;

    if secret_bytes.len() < 32 {
        error!("JWT secret seems too short; please use at least 32 random bytes for HS256");
    }

    // 构造 JWT HS256 EncodingKey
    let jwt_key = EncodingKey::from_secret(&secret_bytes);

    // 读取 RSA 私钥（仅用于 OAEP 解密与导出公钥；与 JWT 密钥完全分离）
    let rsa_key_path =
        env::var("RSA_PRIVATE_KEY_PATH").unwrap_or_else(|_| "rsa_private.pem".into());
    let pem = fs::read_to_string(&rsa_key_path).map_err(|e| {
        error!("failed to read RSA key {}: {}", rsa_key_path, e);
        e
    })?;

    // 解析 RSA 私钥：优先 PKCS#8，失败回退 PKCS#1
    let rsa_private = RsaPrivateKey::from_pkcs8_pem(&pem)
        .or_else(|_| RsaPrivateKey::from_pkcs1_pem(&pem))
        .map_err(|e| {
            error!("failed to parse RSA private key: {}", e);
            e
        })?;

    // 应用状态
    let state = AppState {
        jwt_key: Arc::new(jwt_key),
        rsa_private: Arc::new(rsa_private),
        jwt_exp_minutes: env::var("JWT_EXPIRE_MINUTES")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(60),
        pam_service: env::var("PAM_SERVICE").unwrap_or_else(|_| "flaskapi".into()),
        max_payload_age_secs: env::var("MAX_PAYLOAD_AGE_SECS")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(60),
    };

    // 路由：探活 / 公钥 / 颁发
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/pubkey", get(get_pubkey))
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

/// 返回 RSA 公钥（PEM），用于客户端进行加密
async fn get_pubkey(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    let pub_pem = state
        .rsa_private
        .to_public_key()
        .to_public_key_pem(LineEnding::LF)
        .map_err(|_| ApiError::Internal("export public key"))?;

    Ok((
        StatusCode::OK,
        (
            [("Content-Type", "application/x-pem-file")],
            pub_pem.to_string(),
        ),
    ))
}

/// 解密密文 -> PAM 认证 -> HS256 签 JWT
async fn issue_token(
    State(state): State<AppState>,
    Json(req): Json<EncryptedTokenRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // 1) Base64 解码密文
    let ciphertext = B64
        .decode(&req.ciphertext_b64)
        .map_err(|_| ApiError::BadRequest("invalid base64"))?;

    // 2) RSA-OAEP(SHA-256) 解密
    let oaep = Oaep::new::<Sha256>();
    let plaintext = state
        .rsa_private
        .decrypt(oaep, &ciphertext)
        .map_err(|_| ApiError::BadRequest("invalid ciphertext"))?;

    // 3) 解析明文 JSON
    let payload: AuthPayload = serde_json::from_slice(&plaintext)
        .map_err(|_| ApiError::BadRequest("invalid decrypted json"))?;

    let username = payload.username.trim();
    if username.is_empty() || payload.password.is_empty() {
        return Err(ApiError::BadRequest("username and password are required"));
    }

    // 4) 抗重放：校验 ts
    let now = OffsetDateTime::now_utc();
    let age = now.unix_timestamp() - payload.ts;
    if age < 0 || age > state.max_payload_age_secs {
        return Err(ApiError::BadRequest("stale or future timestamp"));
    }

    // 5) PAM 认证
    let mut auth = Authenticator::with_password(&state.pam_service)
        .map_err(|_| ApiError::Internal("pam init"))?;
    auth.get_handler()
        .set_credentials(username, &payload.password);
    auth.authenticate().map_err(|_| ApiError::AuthFailed)?;

    // 6) HS256 颁发 JWT
    let exp = now + Duration::minutes(state.jwt_exp_minutes);
    let claims = Claims {
        sun: username.to_string(),
        iat: now.unix_timestamp(),
        exp: exp.unix_timestamp(),
    };
    let mut header = Header::new(Algorithm::HS256);

    header.typ = Some("JWT".to_string());

    let token =
        encode(&header, &claims, &state.jwt_key).map_err(|_| ApiError::Internal("jwt encode"))?;

    // 立刻解析头部核对
    let h = decode_header(&token).map_err(|_| ApiError::Internal("decode header"))?;
    assert_eq!(h.alg, Algorithm::HS256);

    // === 可选签发 OpenSSH 用户证书 ===
    let mut ssh_user_cert_text: Option<String> = None;
    let mut ssh_user_cert_exp: Option<i64> = None;

    if let Some(ref ssh_pub) = payload.ssh_pubkey {
        // 粗略校验 OpenSSH 公钥前缀
        let looks_like_ssh_pub = ssh_pub.starts_with("ssh-ed25519 ")
            || ssh_pub.starts_with("ssh-rsa ")
            || ssh_pub.starts_with("sk-");
        if !looks_like_ssh_pub {
            return Err(ApiError::BadRequest("invalid ssh_pubkey"));
        }

        let ca_path = env::var("SSH_CA_KEY_PATH").unwrap_or_else(|_| "/etc/ssh/ssh_user_ca".into());
        let ext_env = env::var("SSH_CERT_EXTENSIONS").unwrap_or_default();
        let extensions: Vec<&str> = ext_env
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        let serial_base = env::var("SSH_CERT_SERIAL_BASE")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        match issue_ssh_user_cert(
            &ca_path,
            username,
            now,
            exp,
            ssh_pub,
            &extensions,
            serial_base,
        ) {
            Ok(cert_txt) => {
                ssh_user_cert_text = Some(cert_txt);
                ssh_user_cert_exp = Some(exp.unix_timestamp());
            }
            Err(e) => {
                // 证书失败不影响 JWT；仅记录日志
                error!("issue ssh cert error: {:?}", e);
            }
        }
    }

    Ok((
        StatusCode::OK,
        Json(TokenResponse {
            access_token: token,
            token_type: "Bearer",
            expires_in: state.jwt_exp_minutes * 60,
            ssh_user_cert: ssh_user_cert_text,
            ssh_user_cert_exp,
        }),
    ))
}
