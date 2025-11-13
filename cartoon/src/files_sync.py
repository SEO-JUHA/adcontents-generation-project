
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()

ALLOWED_ROOTS = [Path("./uploads").resolve(), Path(".").resolve()]

def _is_allowed(path: Path) -> bool:
    try:
        rp = path.resolve()
    except Exception:
        return False
    return any(str(rp).startswith(str(root)) for root in ALLOWED_ROOTS)

@router.get("/get")
async def get_file(path: str = Query(..., description="서버의 절대/상대 경로")):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not _is_allowed(p):
        raise HTTPException(status_code=403, detail="Path not allowed")
    return FileResponse(str(p))
