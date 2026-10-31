"""
    It describes what information the dashboard keeps track of.
    The five tables 
    1. ModelRun:
        ModelRun seeks to answer this questions when a model is being train:
            - what is the model called 
            - which version is this ?
            - Is it still training, done or did it fail?
            - what settings did I use?
            - when did it start and finish?
            - where is the saved file?
            - if it failed, what was the error?

"""

import uuid
from datetime import datetime 

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def _uuid_str() -> str:
    """Return a fresh UUID as a 36-char String."""
    return str(uuid.uuid4())


# model runs
class ModelRun(Base):
    __tablename__ = "model_runs"
    id = Column(String(36), primary_key=True, default=_uuid_str)
    model_name = Column(String(120), nullable=False, index=True)
    version = Column(String(40), nullable=False)
    status = Column(String(20), nullable=False, default="queued", index=True)
    hyperparams = Column(JSON, default=dict)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)
    error = Column(Text)
    artifact_path = Column(String(500))

    metrics = relationship(
        "MetricPoint",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_run_model_status", "model_name", "status"),
    )


    def __repr__(self) -> str:
        return f"<ModelRun {self.model_name} v{self.version} [{self.status}]>"



class MetricPoint(Base):
    __tablename__ = "metric_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        String(36),
        ForeignKey("model_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    epoch = Column(Integer, nullable=False)
    loss = Column(Float)
    accuracy = Column(Float)
    val_loss = Column(Float)
    val_accuracy = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    run = relationship("ModelRun", back_populates="metrics")


class InferenceLog(Base):
    __tablename__ = "inference_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(120), nullable=False, index=True)
    model_version = Column(String(40))
    latency_ms = Column(Float, nullable=False)
    success = Column(Boolean, default=True)
    input_hash = Column(String(64))
    prediction = Column(String(80))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:
        return f"<InferenceLog {self.model_name} {self.latency_ms:.1f}ms"



class DriftSnapshot(Base):
    __tablename__ = "drift_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(120), nullable=False, index=True)
    window_start = Column(DateTime)
    window_end = Column(DateTime)
    psi_score = Column(Float)
    ks_statistic = Column(Float)
    threshold = Column(Float, default=0.2)
    drifted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<DriftSnapshot {self.model_name} psi={self.psi_score:.3f} drifted={self.drifted}>"



class RetrainTrigger(Base):
    __tablename__ = "retrain_triggers"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(120), nullable=False, index=True)
    reason = Column(String(240))
    severity = Column(String(20), default="warning")
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:
        return f"<RetrainTrigger {self.model_name} [{self.severity} ack={self.acknowledged}]>"