# Create an array to store background process IDs
declare -a JOB_IDS
batch_size_sample=50
# Submit five jobs in a loop
inFold_subset=101
seed=50 # 40 30 20 10
for ((i=0; i<inFold_subset; i++)); do
    JOB_NAME="s0sub${i}"
    sbatch --job-name=$JOB_NAME \
           --output="/home/fader/seed${seed}_${JOB_NAME}.out" \
           --error="/home/fader/seed${seed}_${JOB_NAME}.err" \
           --ntasks=1 \
           --cpus-per-task=1 \
           --gres=gpu:1 \
           --partition=gpu \
           --nodelist=fader-sandbox \
           --mem=32GB \
           --time=infinite <<EOF &  # Corrected part here
#!/bin/bash
echo "batch_size_sample in job script: $batch_size_sample"
python -u "/home/fader/Inference_2.py" \
    --weights  "/path/to/GATSegDiff/Dockerfile/GATSegDiff/model/LiVS_reannotated_subset_F1_model160000.pt" \
    --input "/path/to/Task08_HepaticVessel/Task08_HepaticVessel/imagesTr"\
    --output "/path/to/output" \

EOF

    # Get the job ID of the last submitted job
    JOB_IDS+=($!)
done

# Wait for all background jobs to finish
wait "${JOB_IDS[@]}"

    # --dpm_solver True \
    # --num_ensemble 3 \
    # --diffusion_steps 30 
