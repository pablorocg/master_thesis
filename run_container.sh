docker run --rm -dit --gpus all --name tfm_prg -p 8888:8888 \
--memory="15g" --memory-reservation="14g" \
-v /mnt/LVMData/Datasets/Tractoinferno/:/app/dataset/Tractoinferno \
-v /mnt/LVMData/Datasets/HCP105_Zenodo_NewTrkFormat/:/app/dataset/HCP_105 \
-v /home/vvillena/tfm_prg/tfm_prg:/app pablorg:torch_latest /bin/bash
# jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

