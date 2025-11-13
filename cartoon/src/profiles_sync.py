from __future__ import annotations
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# --- DB / Models 가져오기 (backend/database.py를 그대로 사용) ---
try:
    from backend.database import SessionLocal, Logo, FourcutsImage
except Exception as e:
    raise RuntimeError(
        f"[profiles_sync] backend.database import failed: {e}\n"
        "프로젝트 PYTHONPATH 또는 패키지 구조를 확인하세요. (예: uv 실행 경로)"
    )

router = APIRouter(tags=["profiles-sync"])

# Pydantic Schemas
class SaveLogo(BaseModel):
    user_id: int = Field(..., description="users.id")
    logo_path: str

class LogoOut(BaseModel):
    logo_path: str

class RecentImagesOut(BaseModel):
    images: List[str]
    count: int
    created_at: Optional[str] = None

# Helpers
def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API
@router.post("/save-logo")
def save_logo(body: SaveLogo):
    db = next(_get_db())
    try:
        # 기존 로고 삭제 (단일로 유지)
        db.query(Logo).filter(Logo.user_id == body.user_id).delete(synchronize_session=False)
        db.add(Logo(user_id=body.user_id, image_path=body.logo_path))
        db.commit()
        return {"ok": True}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"save_logo failed: {e}")
    finally:
        db.close()


@router.get("/logo", response_model=LogoOut)
def get_logo(user_id: int):
    db = next(_get_db())
    try:
        row = (
            db.query(Logo)
            .filter(Logo.user_id == user_id)
            .order_by(Logo.created_at.desc())
            .first()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        return LogoOut(logo_path=row.image_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_logo failed: {e}")
    finally:
        db.close()

@router.get("/recent-images", response_model=RecentImagesOut)
def get_recent_images(user_id: int, limit: int = 4):
    db = next(_get_db())
    try:
        q = (
            db.query(FourcutsImage)
            .filter(FourcutsImage.user_id == user_id)
            .order_by(FourcutsImage.created_at.desc())
            .limit(int(limit))
        )
        rows = q.all()
        if not rows:
            return RecentImagesOut(images=[], count=0, created_at=None)

        images = []
        created_at = None
        for r in rows:
            p = (r.image_path or "").strip()
            if "panel_" in p and "_base." in p:
                images.append(p)

        if len(images) < len(rows):
            for r in rows:
                if len(images) >= limit:
                    break
                p = (r.image_path or "").strip()
                if p and p not in images:
                    images.append(p)

        created_at = rows[0].created_at.isoformat() if rows and rows[0].created_at else None
        return RecentImagesOut(images=images[:limit], count=len(images[:limit]), created_at=created_at)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_recent_images failed: {e}")
    finally:
        db.close()