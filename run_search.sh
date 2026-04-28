#!/bin/bash

# 1. 基础配置
CONFIG="configs/prcc/vit_ccreid_prcc.yml"
LOG_FILE="search_results.csv"

# 初始化表头 (ID权重, 衣着权重, mAP, Rank1)
echo "ID_W,Cloth_W,mAP,Rank1" > $LOG_FILE

# 定义实验组合
combinations=(
    "0.0 0.0"
    "2.0 0.5"
    "2.0 0.1"
    "1.0 0.1"
    "3.0 0.1"
)

# 自动从 YML 提取输出目录，确保清理时不会删错文件夹
OUT_DIR=$(grep "OUTPUT_DIR" $CONFIG | awk '{print $2}' | tr -d " '\"")
echo ">>> 检测到输出目录: $OUT_DIR"

for combo in "${combinations[@]}"; do
    read -r id_w cloth_w <<< "$combo"
    echo "---------------------------------------------------------"
    echo ">>> 开始实验: ID_Weight=$id_w, Cloth_Weight=$cloth_w"
    
    # 执行训练
    python train.py --config_file $CONFIG MODEL.I2T_ID_WEIGHT $id_w MODEL.I2T_CLOTH_WEIGHT $cloth_w
    
    # 检查训练是否正常结束（生成了 model_60.pth）
    if [ ! -f "$OUT_DIR/model_60.pth" ]; then
        echo "!!! 警告: 未检测到 model_60.pth，训练可能中途崩溃。检查 console.log"
        echo "$id_w,$cloth_w,FAIL,FAIL" >> $LOG_FILE
        continue
    fi

    # 执行测试 (合并 stdout 和 stderr 到临时日志，方便 Debug)
    echo ">>> 正在进行评测..."
    python test_prcc.py --config_file $CONFIG > temp_test.log 2>&1
    
    # 稳健提取数值：忽略中文，直接找关键字，并剔除百分号和空格
    map_val=$(grep "mAP" temp_test.log | head -n 1 | awk -F':' '{print $2}' | tr -d " %")
    rank1_val=$(grep "Rank-1" temp_test.log | head -n 1 | awk -F':' '{print $2}' | tr -d " %")

    # 如果没抓到数值，填入 ERROR 占位
    if [ -z "$map_val" ]; then
        map_val="ERROR"
        rank1_val="ERROR"
        echo "!!! 结果抓取失败。请查看 temp_test.log 获取报错详情"
    fi

    echo "$id_w,$cloth_w,$map_val,$rank1_val" >> $LOG_FILE
    echo ">>> 本轮结果: mAP=$map_val, Rank-1=$rank1_val"

    # 清理物理权重，释放磁盘空间
    echo ">>> 正在清理该目录下的权重文件..."
    rm -f $OUT_DIR/*.pth
done

echo "Done! 结果已汇总至 $LOG_FILE"