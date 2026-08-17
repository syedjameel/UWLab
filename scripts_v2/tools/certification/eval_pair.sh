#!/usr/bin/env bash
# eval_pair.sh <step>  — eval both B variants at given ckpt step on GPU3
S=$1
~/vis_eval.sh 2 B3cam_$S OmniReset-Ur10eLinearGripper-ObjectInBoxPaper-RGB-Play-v0 $HOME/diffusion_policy/data/outputs/ur10e_v2_B_3cam/checkpoints/step_0$S.ckpt
~/vis_eval.sh 2 B2cam_$S OmniReset-Ur10eLinearGripper-ObjectInBoxPaper-RGB-Play-v0 $HOME/diffusion_policy/data/outputs/ur10e_v2_B_2cam/checkpoints/step_0$S.ckpt
echo "### PAIR_DONE $S" >> $HOME/ft_logs/eval_pair.log
