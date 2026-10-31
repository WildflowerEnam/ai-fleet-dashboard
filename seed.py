import random 
from datetime import datetime, timedelta

from shared.db import SessionLocal 
from shared.models import (
    DriftSnapshot,
    InferenceLog,
    MetricPoint,
    ModelRun,
    RetrainTrigger,
)


MODEL_NAME = "cliniflow-triage-v1"
MODEL_VERSION = "1.0"


def main() -> None:
    now = datetime.utcnow()

    with SessionLocal() as session:
        run = ModelRun(
            model_name=MODEL_NAME,
            version=MODEL_VERSION,
            status="completed",
            hyperparams={"C":1.0, "max_iter":200, "solver": "saga"},
            started_at=now - timedelta(minutes=5),
            finished_at=now - timedelta(minutes=1),
            artifact_path=f"/models/{MODEL_NAME}--{MODEL_VERSION}.joblib",
        )
        session.add(run)
        session.flush()
        loss_curve = [
            0.68, 0.52, 0.41, 0.34, 0.29,
            0.25, 0.22, 0.20, 0.18, 0.17,
        ]

        acc_curve = [
            0.62, 0.72, 0.80, 0.85, 0.88,
            0.90, 0.91, 0.92, 0.93, 0.94,
        ]

        for epoch, (loss, acc) in enumerate(zip(loss_curve, acc_curve), start=1):
            session.add(MetricPoint(
                run_id=run.id,
                epoch=epoch,
                loss=loss,
                accuracy=acc,
                val_loss=loss + 0.03,
                val_accuracy=acc - 0.02,
            ))


        
        for i in range(15):
            session.add(InferenceLog(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                latency_ms=round(random.uniform(20.0, 80.0), 1),
                success=True,
                input_hash=f"{random.getrandbits(64):016x}",
                prediction=random.choice(["0", "1"]),
                timestamp=now - timedelta(seconds=5 * (15 - i))
            ))


        session.add(DriftSnapshot(
            model_name=MODEL_NAME,
            window_start=now - timedelta(hours=1),
            window_end=now,
            psi_score=0.31,
            ks_statistic=0.44,
            threshold=0.2,
            drifted=True,
        ))


        session.add(RetrainTrigger(
            model_name=MODEL_NAME,
            reason="Drift detected (PSI=0.31, KS=0.44)",
            severity="warning",
            acknowledged=False,
        ))

        session.commit()

    
    print("Seeded:")
    print(" 1 model run")
    print(f"    {len(loss_curve)} metric points")
    print("  15 inference logs")
    print("     1 drift snapshot")
    print(" 1 retrain trigger")



if __name__ == "__main__":
    main()