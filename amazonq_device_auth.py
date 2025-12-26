"""
Amazon Q Device Authorization Flow Handler
AWS SSO OIDC Device Authorization for web-based login
"""
import httpx
import uuid
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# AWS SSO OIDC endpoint
SSO_OIDC_ENDPOINT = "https://oidc.us-east-1.amazonaws.com"

# In-memory session storage
_sessions: Dict[str, 'DeviceAuthSession'] = {}


@dataclass
class DeviceAuthSession:
    session_id: str
    client_id: str
    client_secret: str
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_at: datetime
    interval: int
    status: str = "pending"
    error: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None


async def register_client() -> Dict[str, str]:
    """Register OIDC client with AWS SSO"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{SSO_OIDC_ENDPOINT}/client/register",
            json={
                "clientName": "amq2api-web-login",
                "clientType": "confidential",
                "scopes": [
                    "codewhisperer:completions",
                    "codewhisperer:analysis",
                    "codewhisperer:conversations",
                ]
            },
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        return {
            "clientId": data["clientId"],
            "clientSecret": data["clientSecret"]
        }


async def start_device_authorization(client_id: str, client_secret: str) -> Dict[str, Any]:
    """Start device authorization flow"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{SSO_OIDC_ENDPOINT}/device_authorization",
            json={
                "clientId": client_id,
                "clientSecret": client_secret,
                "startUrl": "https://view.awsapps.com/start"
            },
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()


async def poll_for_token(client_id: str, client_secret: str, device_code: str) -> Dict[str, Any]:
    """Poll for access token using device code"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{SSO_OIDC_ENDPOINT}/token",
            json={
                "clientId": client_id,
                "clientSecret": client_secret,
                "deviceCode": device_code,
                "grantType": "urn:ietf:params:oauth:grant-type:device_code"
            },
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 400:
            error_data = response.json()
            error = error_data.get("error", "unknown_error")
            if error == "authorization_pending":
                return {"status": "pending"}
            elif error == "slow_down":
                return {"status": "slow_down"}
            elif error == "expired_token":
                return {"status": "expired", "error": "Device code expired"}
            else:
                return {"status": "error", "error": error_data.get("error_description", error)}

        response.raise_for_status()
        data = response.json()
        return {
            "status": "completed",
            "accessToken": data["accessToken"],
            "refreshToken": data.get("refreshToken"),
            "expiresIn": data.get("expiresIn")
        }


async def create_session() -> DeviceAuthSession:
    """Create new device authorization session"""
    client_data = await register_client()
    device_data = await start_device_authorization(client_data["clientId"], client_data["clientSecret"])

    session_id = str(uuid.uuid4())
    session = DeviceAuthSession(
        session_id=session_id,
        client_id=client_data["clientId"],
        client_secret=client_data["clientSecret"],
        device_code=device_data["deviceCode"],
        user_code=device_data["userCode"],
        verification_uri=device_data["verificationUri"],
        verification_uri_complete=device_data["verificationUriComplete"],
        expires_at=datetime.now() + timedelta(seconds=device_data.get("expiresIn", 600)),
        interval=device_data.get("interval", 5)
    )

    _sessions[session_id] = session
    cleanup_expired_sessions()
    return session


def get_session(session_id: str) -> Optional[DeviceAuthSession]:
    """Get session by ID"""
    return _sessions.get(session_id)


async def poll_session(session_id: str) -> Dict[str, Any]:
    """Poll for token completion and import account on success"""
    session = get_session(session_id)
    if not session:
        return {"status": "error", "error": "Session not found"}

    if session.status != "pending":
        return {"status": session.status, "error": session.error}

    if datetime.now() > session.expires_at:
        session.status = "error"
        session.error = "Device code expired"
        return {"status": "error", "error": "Device code expired"}

    result = await poll_for_token(session.client_id, session.client_secret, session.device_code)

    if result["status"] == "completed":
        session.status = "completed"
        session.access_token = result["accessToken"]
        session.refresh_token = result.get("refreshToken")

        # Import account to database
        from account_manager import create_account
        create_account(
            label=f"AmazonQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            client_id=session.client_id,
            client_secret=session.client_secret,
            refresh_token=session.refresh_token,
            access_token=session.access_token,
            enabled=True,
            account_type="amazonq"
        )
        logger.info(f"Amazon Q account imported via device auth: {session_id}")
        return {"status": "completed"}

    elif result["status"] in ("error", "expired"):
        session.status = "error"
        session.error = result.get("error")
        return result

    return {"status": "pending"}


def cleanup_expired_sessions():
    """Remove expired sessions"""
    now = datetime.now()
    expired = [sid for sid, s in _sessions.items() if now > s.expires_at + timedelta(minutes=5)]
    for sid in expired:
        del _sessions[sid]
