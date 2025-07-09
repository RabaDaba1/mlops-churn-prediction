from kfp import compiler
from kfp.dsl import ContainerSpec, container_component, pipeline


@container_component
def dvc_repro_op():
    return ContainerSpec(
        image="yourdockerhubusername/churn-prediction-api:latest",  # Use the API image which has all dependencies
        command=["sh", "-c"],
        args=[
            """
            set -ex
            git config --global --add safe.directory /workspace
            dvc remote modify b2remote access_key_id $B2_ACCESS_KEY_ID
            dvc remote modify b2remote secret_access_key $B2_SECRET_ACCESS_KEY
            wandb login $WANDB_API_KEY
            dvc pull -r b2remote
            dvc repro -f
            dvc push -r b2remote
            # Commit and push dvc.lock - requires git credentials setup in cluster
            # git config --local user.email "action@github.com"
            # git config --local user.name "GitHub Action"
            # git add dvc.lock
            # if ! git diff --staged --quiet; then
            #   git commit -m "Update dvc.lock from Kubeflow pipeline"
            #   git push
            # fi
            """
        ],
    )


@pipeline(
    name="ML Retraining Pipeline",
    description="A pipeline to retrain the churn prediction model.",
)
def ml_retraining_pipeline():
    dvc_repro_task = dvc_repro_op().set_display_name("Run DVC Pipeline")
    # Secrets need to be created in the Kubeflow namespace
    # e.g., kubectl create secret generic b2-creds --from-literal=key=...
    # e.g., kubectl create secret generic wandb-creds --from-literal=key=...
    dvc_repro_task.container.add_env_variable_from_secret(
        "B2_ACCESS_KEY_ID", "b2-credentials", "B2_ACCESS_KEY_ID"
    ).add_env_variable_from_secret(
        "B2_SECRET_ACCESS_KEY", "b2-credentials", "B2_SECRET_ACCESS_KEY"
    ).add_env_variable_from_secret(
        "WANDB_API_KEY", "wandb-credentials", "WANDB_API_KEY"
    ).add_env_variable_from_secret(
        "WANDB_PROJECT", "wandb-credentials", "WANDB_PROJECT"
    ).add_env_variable_from_secret(
        "WANDB_ENTITY", "wandb-credentials", "WANDB_ENTITY"
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ml_retraining_pipeline,
        package_path="kubeflow_pipeline.yaml",
    )
    print("Kubeflow pipeline compiled to kubeflow_pipeline.yaml")
