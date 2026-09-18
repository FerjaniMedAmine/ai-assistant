from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import Memory
from backend.schemas import MemoryCreate, MemoryResponse

router = APIRouter(prefix="/api/memory", tags=["memory"])

@router.get("", response_model=List[MemoryResponse])
def list_memories(db: Session = Depends(get_db)):
    """
    Retrieve all long-term persistent user facts.
    """
    memories = db.query(Memory).order_by(Memory.created_at.desc()).all()
    return memories

@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
def create_memory(payload: MemoryCreate, db: Session = Depends(get_db)):
    """
    Manually store a persistent fact about the user.
    """
    mem = Memory(content=payload.content.strip())
    db.add(mem)
    db.commit()
    db.refresh(mem)
    return mem

@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(memory_id: str, db: Session = Depends(get_db)):
    """
    Permanently delete a persistent fact from PostgreSQL.
    """
    mem = db.query(Memory).filter(Memory.id == memory_id).first()
    if not mem:
        raise HTTPException(status_code=404, detail="Memory item not found")

    db.delete(mem)
    db.commit()
    return None
