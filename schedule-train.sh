#!/bin/bash

function perform_in_time() {   
    start_time=$1 
    echo "scheduel in $start_time"
    # 计算距离开始的时间
    wait_time=$(( $(date -d "$start_time" +%s) - $(date +%s) ))

    RESET=$(echo -en '\033[0m')
    COLOR=$(echo -en '\033[01;35m')

    # 统计还剩下多长时间并输出
    while [ $wait_time -gt 0 ]
    do
        # 该行代码用于清空终端输出并将光标定位到第一列，以便实现文字动态更新的效果
        printf "\033c"
        
        # 计算还剩下多少小时、分钟、秒
        hours=$((${wait_time}/3600))
        minutes=$(((${wait_time}%3600)/60))
        seconds=$((${wait_time}%60))
        
        # 输出倒计时并等待1秒
        printf "\e[1mStart in ${COLOR}%2d${RESET}h ${COLOR}%2d${RESET}min ${COLOR}%2d${RESET}s\e[0m\n" $hours $minutes $seconds
        sleep 1
        
        # 更新等待的时间
        wait_time=$(($wait_time - 1))
    done
    clear
}


perform_in_time "08:00"

eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt



export CUDA_VISIBLE_DEVICES=0,1


python train-classifier.py