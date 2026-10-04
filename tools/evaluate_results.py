"""
===========================================
CityAVOS AAAI26 Evaluation Tool

2026-09-30 新增

功能:
1. 读取每个task实验日志
2. 计算论文指标:
   SR
   SPL
   MSS
   NE

保留原作者代码:
无

===========================================
"""


import os
import json
import numpy as np


# ===============================
# 路径配置
# ===============================

RESULT_DIR = (
    "../output/experiemnt_data/ours"
)


SUMMARY_FILE = (
    "../output/experiemnt_data/ours/"
    "experiment_summary.json"
)



# ===============================
# 计算轨迹长度
# ===============================

def trajectory_length(points):

    length = 0


    for i in range(len(points)-1):

        p1=np.array(points[i])

        p2=np.array(points[i+1])


        length += np.linalg.norm(
            p2-p1
        )


    return length



# ===============================
# 欧式距离
# ===============================

def distance(a,b):

    return np.linalg.norm(
        np.array(a)-np.array(b)
    )



# ===============================
# 单task计算
# ===============================

def evaluate_task(data):


    success = data["success"]


    steps = data["steps"]


    trajectory = data["trajectory"]


    target = data["target_position"]


    final_pos = data["final_position"]



    # 实际距离

    path_length = trajectory_length(
        trajectory
    )


    # 最短距离

    shortest_distance = distance(
        trajectory[0],
        target
    )


    # 最终距离

    final_distance = distance(
        final_pos,
        target
    )


    # ===================
    # MSS
    # ===================

    mss = steps



    # ===================
    # NE
    # ===================

    ne = final_distance



    # ===================
    # SPL
    # ===================

    if success:

        spl = (
            shortest_distance /
            max(
                path_length,
                shortest_distance
            )
        )

    else:

        spl = 0



    return {

        "success":success,

        "steps":steps,

        "path_length":path_length,

        "shortest_distance":
        shortest_distance,

        "MSS":mss,

        "NE":ne,

        "SPL":spl

    }




# ===============================
# 主函数
# ===============================

def main():


    files=[]


    for f in os.listdir(RESULT_DIR):

        if (
            f.startswith("data_")
            and f.endswith(".json")
        ):

            files.append(f)



    files.sort(
        key=lambda x:int(
            x.split("_")[1]
            .split(".")[0]
        )
    )



    results=[]


    for f in files:


        path=os.path.join(
            RESULT_DIR,
            f
        )


        with open(
            path,
            "r",
            encoding="utf-8"
        ) as fp:


            data=json.load(fp)



        result=evaluate_task(data)


        result["task_file"]=f


        results.append(result)



    total=len(results)



    success_num=sum(
        r["success"]
        for r in results
    )


    SR = success_num / total



    SPL=np.mean(
        [
            r["SPL"]
            for r in results
        ]
    )



    MSS=np.mean(
        [
            r["MSS"]
            for r in results
        ]
    )


    NE=np.mean(
        [
            r["NE"]
            for r in results
        ]
    )



    summary={


        "total_tasks":
        total,


        "success_tasks":
        success_num,


        "failure_tasks":
        total-success_num,


        "metrics":{


            "SR":
            SR,


            "SPL":
            SPL,


            "MSS":
            MSS,


            "NE":
            NE

        },


        "detail":
        results

    }



    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )


    print("====================")

    print("Experiment Finished")

    print("====================")

    print(
        "Tasks:",
        total
    )

    print(
        "Success:",
        success_num
    )


    print(
        "SR:",
        SR
    )

    print(
        "SPL:",
        SPL
    )

    print(
        "MSS:",
        MSS
    )

    print(
        "NE:",
        NE
    )



if __name__=="__main__":

    main()