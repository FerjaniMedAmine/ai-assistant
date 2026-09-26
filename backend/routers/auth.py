from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from backend.config import config
from backend.database import get_db
from backend.models import User

router = APIRouter(prefix="/api/auth", tags=["auth"])
oauth = OAuth()
if config.GOOGLE_OAUTH_CLIENT_ID and config.GOOGLE_OAUTH_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=config.GOOGLE_OAUTH_CLIENT_ID,
        client_secret=config.GOOGLE_OAUTH_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Sign in with Google to continue")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Session expired; sign in again")
    return user


@router.get("/login")
async def login(request: Request):
    if not oauth.google:
        raise HTTPException(status_code=503, detail="Google OAuth is not configured")
    request.session.clear()
    return await oauth.google.authorize_redirect(request, config.GOOGLE_OAUTH_REDIRECT_URI)


@router.get("/callback")
async def callback(request: Request, db: Session = Depends(get_db)):
    if not oauth.google:
        raise HTTPException(status_code=503, detail="Google OAuth is not configured")
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        raise HTTPException(status_code=401, detail="Google sign-in failed")
    profile = token.get("userinfo") or {}
    if not profile.get("sub") or not profile.get("email") or not profile.get("email_verified"):
        raise HTTPException(status_code=401, detail="A verified Google account is required")

    user = db.query(User).filter(User.google_sub == profile["sub"]).first()
    if not user:
        user = User(google_sub=profile["sub"], email=profile["email"], name=profile.get("name") or profile["email"])
        db.add(user)
    else:
        user.email = profile["email"]
        user.name = profile.get("name") or profile["email"]
    user.picture = profile.get("picture")
    db.commit()
    db.refresh(user)
    request.session.clear()
    request.session["user_id"] = user.id
    return RedirectResponse(config.FRONTEND_ORIGIN, status_code=303)


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "picture": user.picture}


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"status": "signed_out"}
